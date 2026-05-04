import math

import torch
from torch.optim import Optimizer

__all__ = ["ScionC"]


class ScionC(Optimizer):
    """
    Minimal ScionC optimizer.

    Each group supplies a ULMO. The default constrained SCG update is:

        m <- beta * m + (1 - beta) * grad
        p <- zeta * p + eta * ulmo(m)

    The additive scale `eta` is taken from `lr` (step-scale form). If
    `rms_solve` and `rms_radius` are set on a group, `eta` is dynamically
    solved via the one-step RMS solve and capped by `lr`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        memory_beta: float = 0.95,
        readout_mu: float = 1.0,
        ulmo=None,
        weight_retention: float = 1.0,
        rms_radius: float | None = None,
        rms_solve: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not (0.0 <= memory_beta < 1.0):
            raise ValueError(f"invalid memory_beta: {memory_beta}")
        if not (0.0 <= readout_mu <= 1.0):
            raise ValueError(f"invalid readout_mu: {readout_mu}")
        if not (0.0 < weight_retention <= 1.0):
            raise ValueError(f"invalid weight_retention: {weight_retention}")

        super().__init__(
            params,
            dict(
                lr=lr,
                memory_beta=memory_beta,
                readout_mu=readout_mu,
                ulmo=ulmo,
                weight_retention=weight_retention,
                rms_radius=rms_radius,
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

            rms_radius = group.get("rms_radius") if group.get("rms_solve") else None
            rms_targets = group.get("rms_targets") if group.get("rms_solve") else None
            lr = float(group["lr"])
            zeta = float(group["weight_retention"])
            if lr == 0.0:
                if zeta != 1.0:
                    for _, p, _ in entries:
                        p.mul_(zeta)
                continue

            updates = self._updates(group["ulmo"], entries)
            for (param_index, p, _), u in zip(entries, updates, strict=True):
                target_radius = None
                if rms_targets is not None:
                    target_radius = float(rms_targets[param_index])
                elif rms_radius is not None:
                    target_radius = float(rms_radius)

                if target_radius is not None:
                    s = (p * u).mean().item()
                    w_sq = p.square().mean().item()
                    v_sq = max(u.square().mean().item(), 1e-15)
                    r_sq = target_radius * target_radius
                    d = zeta * zeta * s * s - v_sq * (zeta * zeta * w_sq - r_sq)
                    if d >= 0.0:
                        root = math.sqrt(d)
                        roots = [
                            value
                            for value in (
                                (-zeta * s - root) / v_sq,
                                (-zeta * s + root) / v_sq,
                            )
                            if value >= 0.0
                        ]
                        eta = min(roots) if roots else 0.0
                    else:
                        eta = max(0.0, -zeta * s / v_sq)
                    eta = min(eta, lr)
                else:
                    eta = lr

                if zeta != 1.0:
                    p.mul_(zeta)
                if eta != 0.0:
                    p.add_(u, alpha=eta)

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
