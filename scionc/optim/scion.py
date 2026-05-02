import torch
from torch.optim import Optimizer

__all__ = ["ScionC"]


class ScionC(Optimizer):
    """
    Minimal ScionC optimizer.

    Each group supplies a ULMO. The step uses action
    coordinates directly:

        p <- shrink * p + eta * ulmo(readout)

    The additive scale is stored in `lr` to match PyTorch optimizer
    conventions. `memory_beta` is the EMA retention and `readout_mu` is the
    Nesterov readout blend between the current gradient and stored EMA.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        memory_beta: float = 0.95,
        readout_mu: float = 1.0,
        ulmo=None,
        shrink: float = 1.0,
    ):
        if lr <= 0.0:
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
            updates = self._updates(group["ulmo"], entries)
            for (p, _), u in zip(entries, updates, strict=True):
                p.mul_(group["shrink"]).add_(u, alpha=group["lr"])

        return loss

    def _collect_entries(self, group) -> list[tuple[torch.Tensor, torch.Tensor]]:
        memory_beta = group["memory_beta"]
        readout_mu = group["readout_mu"]
        entries = []
        for p in group["params"]:
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
            entries.append((p, g))
        return entries

    def _updates(
        self, ulmo, entries: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> list[torch.Tensor]:
        if ulmo is None:
            raise ValueError("ScionC requires a ULMO for every parameter group")

        batch_dir = getattr(ulmo, "batch", None)
        if batch_dir is not None:
            return batch_dir([g for _, g in entries], [p for p, _ in entries])

        set_ulmo_param = getattr(ulmo, "set_param", None)
        updates = []
        for p, g in entries:
            if set_ulmo_param is not None:
                set_ulmo_param(p)
            updates.append(ulmo(g))
        return updates
