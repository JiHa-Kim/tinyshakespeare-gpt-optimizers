import torch
from torch.optim import Optimizer

__all__ = ["ScionC"]


def _rms_solved_group_eta(
    params: list[torch.Tensor],
    updates: list[torch.Tensor],
    shrink: float,
    target_rms: float,
    lr: float,
) -> torch.Tensor:
    total = sum(p.numel() for p in params)
    if total <= 0:
        return torch.zeros((), device=updates[0].device, dtype=updates[0].dtype)

    s = sum(
        (p * u).sum(dtype=torch.float32)
        for p, u in zip(params, updates, strict=True)
    )
    w_sq = torch.stack(torch._foreach_norm(params)).square().sum()
    v_sq = torch.stack(torch._foreach_norm(updates)).square().sum().clamp_min(1e-15)
    s = s / total
    w_sq = w_sq / total
    v_sq = v_sq / total
    target_sq = target_rms * target_rms
    shrink_sq = shrink * shrink
    d = shrink_sq * s * s - v_sq * (shrink_sq * w_sq - target_sq)

    root = d.clamp_min(0.0).sqrt()
    eta = (-shrink * s + root) / v_sq
    alternate = (-shrink * s - root) / v_sq
    eta = torch.where((shrink_sq * w_sq > target_sq) & (s < 0.0), alternate, eta)
    eta = torch.where(d < 0.0, -shrink * s / v_sq, eta)
    return eta.clamp(0.0, lr)


class ScionC(Optimizer):
    """
    Minimal ScionC optimizer.

    Each group supplies a ULMO. The default constrained SCG update is:

        m <- beta * m + (1 - beta) * grad
        p <- shrink * p + eta * ulmo(m)

    The additive scale `eta` is taken from `lr` (step-scale form). If
    `rms_solve` and `target_rms` are set on a group, `eta` is dynamically
    solved via the one-step RMS solve and capped by `lr`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        memory_beta: float = 0.95,
        readout_mu: float = 1.0,
        ulmo=None,
        shrink: float = 1.0,
        target_rms: float | None = None,
        rms_solve: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not (0.0 <= memory_beta < 1.0):
            raise ValueError(f"invalid memory_beta: {memory_beta}")
        if not (0.0 <= readout_mu <= 1.0):
            raise ValueError(f"invalid readout_mu: {readout_mu}")
        if not (0.0 < shrink <= 1.0):
            raise ValueError(f"invalid shrink: {shrink}")

        super().__init__(
            params,
            dict(
                lr=lr,
                memory_beta=memory_beta,
                readout_mu=readout_mu,
                ulmo=ulmo,
                shrink=shrink,
                target_rms=target_rms,
                rms_solve=rms_solve,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            entries = self._collect_entries(group)
            if not entries:
                continue

            target_rms = group.get("target_rms") if group.get("rms_solve") else None
            lr = float(group["lr"])
            shrink = float(group["shrink"])
            if lr == 0.0:
                if shrink != 1.0:
                    for _, p, _ in entries:
                        p.mul_(shrink)
                continue

            updates = self._updates(group["ulmo"], entries)
            params = [p for _, p, _ in entries]
            if target_rms is None:
                if shrink != 1.0:
                    torch._foreach_mul_(params, shrink)
                torch._foreach_add_(params, updates, alpha=lr)
            else:
                eta = _rms_solved_group_eta(
                    params, updates, shrink, float(target_rms), lr
                )
                if shrink != 1.0:
                    torch._foreach_mul_(params, shrink)
                torch._foreach_mul_(updates, eta)
                torch._foreach_add_(params, updates)

        return loss

    def _collect_entries(
        self, group
    ) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        memory_beta = group["memory_beta"]
        readout_mu = group["readout_mu"]
        entries = []
        for param_index, p in enumerate(group["params"]):
            g = p.grad
            if g is None:
                continue
            if g.is_sparse:
                raise RuntimeError("ScionC does not support sparse gradients")

            state = self.state[p]
            if not state:
                state["m"] = torch.zeros_like(
                    g, memory_format=torch.preserve_format
                )

            m = state["m"]
            m.lerp_(g, 1.0 - memory_beta)
            if readout_mu == 1.0:
                g.copy_(m)
            elif readout_mu != 0.0:
                g.lerp_(m, readout_mu)
            entries.append((param_index, p, g))
        return entries

    def _updates(
        self, ulmo, entries: list[tuple[int, torch.Tensor, torch.Tensor]]
    ) -> list[torch.Tensor]:
        if ulmo is None:
            raise ValueError("ScionC requires a ULMO for every parameter group")

        batch_dir = getattr(ulmo, "batch", None)
        if batch_dir is not None:
            return batch_dir([g for _, _, g in entries], [p for _, p, _ in entries])

        set_ulmo_param = getattr(ulmo, "set_param", None)
        updates = []
        for _, p, g in entries:
            if set_ulmo_param is not None:
                set_ulmo_param(p)
            updates.append(ulmo(g))
        return updates
