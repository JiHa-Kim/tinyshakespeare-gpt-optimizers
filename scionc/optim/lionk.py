import torch
from torch.optim import Optimizer

__all__ = [
    "sign_",
    "lionk_S",
    "corrected_eta",
    "LionKCCWDPA",
]


def sign_(x: torch.Tensor) -> torch.Tensor:
    return x.sign_().neg_()


def lionk_S(beta1: float, beta2: float, nesterov: bool = True) -> float:
    beta_eff = beta1 * beta2 if nesterov else beta1
    return (1.0 + beta2) / (
        ((1.0 - beta_eff) ** 2) * (1.0 + beta2) + (beta_eff * beta_eff) * (1.0 - beta2)
    )


def corrected_eta(
    lr: float,
    theta2: float,
    cu2: float = 1.0,
    S: float = 1.0,
    q: float = 1.0,
    eps: float = 1e-12,
) -> float:
    return (lr * lr * cu2 * S) / (2.0 * max(q, eps) * max(theta2, eps))


class LionKCCWDPA(Optimizer):
    """
    Lion-K with corrected decoupled decay, optional cautious masking,
    optional primal averaging, and direct multiplicative shrinkage.

    Parameters are kept at the gradient-eval point
        y_t = (1 - phi) z_t + phi x_t.

    `dir_fn` maps the momentum proxy to a negative update direction.
    `eta` applies fixed decoupled decay; otherwise it is derived from `theta2`.
    `shrink` applies direct multiplicative shrinkage and takes precedence over
    LR-scaled decay.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas=(0.9, 0.99),
        dir_fn=sign_,
        phi: float = 0.0,
        eta: float | None = None,
        shrink: float | None = None,
        theta2: float | None = None,
        cu2: float = 1.0,
        S: float | None = None,
        q: float = 1.0,
        cwd: bool = False,
        nesterov: bool = True,
        eps: float = 1e-12,
    ):
        if lr <= 0.0:
            raise ValueError(f"invalid lr: {lr}")
        beta1, beta2 = betas
        if not (0.0 <= beta1 <= 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"invalid betas: {betas}")
        if not (0.0 <= phi <= 1.0):
            raise ValueError(f"invalid phi: {phi}")
        if shrink is not None and not (0.0 < shrink <= 1.0):
            raise ValueError(f"invalid shrink: {shrink}")

        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                dir_fn=dir_fn,
                phi=phi,
                eta=eta,
                shrink=shrink,
                theta2=theta2,
                cu2=cu2,
                S=S,
                q=q,
                cwd=cwd,
                nesterov=nesterov,
                eps=eps,
                _pa_denom=0.0,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            dir_fn = group["dir_fn"]
            phi = group["phi"]
            cwd = group["cwd"]
            nesterov = group["nesterov"]
            set_dir_param = getattr(dir_fn, "set_param", None)
            batch_dir = getattr(dir_fn, "batch", None)

            shrink = group.get("shrink")
            if shrink is not None:
                shrink = float(shrink)
                if not (0.0 < shrink <= 1.0):
                    raise ValueError(f"invalid shrink: {shrink}")
                if cwd:
                    raise ValueError("direct shrink does not support cautious decay")

            if phi:
                group["_pa_denom"] += lr * lr
                c = (lr * lr) / group["_pa_denom"]
            else:
                c = 0.0

            eta = group["eta"]
            theta2 = group["theta2"]
            if shrink is None and eta is None:
                if theta2 is None:
                    eta = lr * group.get("weight_decay", 0.0)
                else:
                    S = group["S"]
                    if S is None:
                        S = group["S"] = lionk_S(beta1, beta2, nesterov=nesterov)
                    eta = corrected_eta(
                        lr=lr,
                        theta2=theta2,
                        cu2=group["cu2"],
                        S=S,
                        q=group["q"] if cwd else 1.0,
                        eps=group["eps"],
                    )

            entries = []
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("LionKCCWDPA does not support sparse gradients")

                state = self.state[p]
                if not state:
                    state["m"] = g.detach().clone(memory_format=torch.preserve_format)
                    state["z"] = p.detach().clone(memory_format=torch.preserve_format)
                    if phi:
                        state["x"] = p.detach().clone(
                            memory_format=torch.preserve_format
                        )

                m = state["m"]
                z = state["z"]

                # Reuse grad storage as scratch to avoid per-step allocations.
                if nesterov:
                    m.lerp_(g, 1.0 - beta2)
                    if beta1 == 1.0:
                        g.copy_(m)
                    elif beta1 != 0.0:
                        g.lerp_(m, beta1)
                else:
                    if beta1 != 0.0:
                        tmp = state.get("tmp")
                        if tmp is None:
                            tmp = state["tmp"] = m.detach().clone(
                                memory_format=torch.preserve_format
                            )
                        else:
                            tmp.copy_(m)
                    m.lerp_(g, 1.0 - beta2)
                    if beta1 == 1.0:
                        g.copy_(tmp)
                    elif beta1 != 0.0:
                        g.mul_(1.0 - beta1).add_(tmp, alpha=beta1)

                entries.append((p, g, z, state))

            if batch_dir is None:
                updates = []
                for p, g, _, _ in entries:
                    if set_dir_param is not None:
                        set_dir_param(p)
                    updates.append(dir_fn(g))
            else:
                updates = batch_dir(
                    [g for _, g, _, _ in entries],
                    [p for p, _, _, _ in entries],
                )

            for (p, _, z, state), u in zip(entries, updates, strict=True):
                if shrink is not None:
                    z.mul_(shrink)
                elif eta:
                    if cwd:
                        z.addcmul_(p, (p * u > 0).to(dtype=p.dtype), value=-eta)
                    else:
                        z.add_(p, alpha=-eta)

                z.add_(u, alpha=lr)

                if phi:
                    x = state["x"]
                    x.lerp_(z, c)
                    if phi == 1.0:
                        p.copy_(x)
                    else:
                        p.copy_(z).lerp_(x, phi)
                else:
                    p.copy_(z)

        return loss
