import math

import torch
import torch.nn as nn

from scionc.lmos.streaming_svd import HiddenSVDFilterLMO, StreamingSVDSpectralLMO
from scionc.optim.lionk import LionKCCWDPA

__all__ = [
    "ColNormLMO",
    "RowNormLMO",
    "GramNewtonSchulzLMO",
    "SpectralLMO",
    "StreamingSVDSpectralLMO",
    "HiddenSVDFilterLMO",
    "SignLMO",
    "init_colnorm_",
    "init_rownorm_",
    "init_spectral_",
    "init_sign_",
    "init_semiorthogonal_",
    "ScionC",
]


_GNS_COEFFS = (
    (
        7.892582874424408,
        -20.38301394587957,
        13.555306149406924,
    ),
    (
        3.911484868135431,
        -2.5464635929060884,
        0.4268988319673074,
    ),
    (
        3.760657955697423,
        -2.512819018216563,
        0.4323647349070073,
    ),
    (
        3.160399673686287,
        -2.149649518898498,
        0.3996366907664389,
    ),
    (
        2.150183725287622,
        -1.414742461304243,
        0.3220191461514592,
    ),
)
_GNS_RESETS = frozenset({2})


def _gns_coeff(i: int) -> tuple[float, float, float]:
    return _GNS_COEFFS[i if i < len(_GNS_COEFFS) else -1]


def _gns_work_dtype(x: torch.Tensor, work_dtype: torch.dtype | None) -> torch.dtype:
    if work_dtype is not None:
        return work_dtype
    return torch.float16 if x.is_cuda else torch.float32


def _moment2_beta(gram: torch.Tensor, eps: float, safety: float) -> torch.Tensor:
    n = gram.size(-1)
    t1 = (
        gram.diagonal(dim1=-2, dim2=-1)
        .sum(-1, dtype=torch.float32)
        .clamp_min(eps)
    )
    r2 = gram.square().sum(dim=(-2, -1), dtype=torch.float32)
    m2 = r2.div(t1.square()).clamp_min(1.0 / n)
    beta = 1.0 / n + torch.sqrt(((n - 1.0) / n) * (m2 - 1.0 / n))
    return beta.mul(safety).clamp_min(eps)


def _moment2_gram_scale(
    x: torch.Tensor, gram: torch.Tensor, eps: float, safety: float
) -> tuple[torch.Tensor, torch.Tensor]:
    beta = _moment2_beta(gram, eps, safety)
    inv = torch.rsqrt(beta).reshape(-1, 1, 1)
    inv_x = inv.to(x.dtype)
    return gram * inv_x.square(), inv_x


def _gram_newton_schulz_core(
    x: torch.Tensor, steps: int, eps: float, bound_safety: float
) -> torch.Tensor:
    gram = torch.bmm(x, x.mT)
    gram, x_scale = _moment2_gram_scale(x, gram, eps, bound_safety)
    eye = torch.eye(gram.size(-1), dtype=x.dtype, device=x.device).expand_as(gram)
    q = None

    for i in range(steps):
        a, b, c = _gns_coeff(i)
        if i in _GNS_RESETS and q is not None:
            if x_scale is not None:
                q = q * x_scale
                x_scale = None
            x = torch.bmm(q, x)
            gram = torch.bmm(x, x.mT)
            q = None

        z = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        q = z + a * eye if q is None else torch.baddbmm(q, q, z, beta=a)

        if i < steps - 1 and i + 1 not in _GNS_RESETS:
            rz = torch.baddbmm(gram, gram, z, beta=a)
            gram = torch.baddbmm(rz, z, rz, beta=a)

    if q is None:
        return x
    if x_scale is not None:
        q = q * x_scale
    return torch.bmm(q, x)


def _standard_newton_schulz_core(
    x: torch.Tensor, steps: int, eps: float, bound_safety: float
) -> torch.Tensor:
    gram = torch.bmm(x, x.mT)
    gram, x_scale = _moment2_gram_scale(x, gram, eps, bound_safety)
    for i in range(steps):
        a, b, c = _gns_coeff(i)
        if i:
            gram = torch.bmm(x, x.mT)
        update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, update, x, beta=a)
        if x_scale is not None:
            x = x * x_scale
            x_scale = None
    return x


def gram_newton_schulz_uvt(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    work_dtype: torch.dtype | None = None,
    bound_safety: float = 1.05,
) -> torch.Tensor:
    if g.ndim < 2:
        raise ValueError("gram_newton_schulz_uvt expects a matrix or batch of matrices")

    original_shape = g.shape
    original_dtype = g.dtype
    x = g.reshape(-1, *g.shape[-2:]) if g.ndim > 3 else g
    if x.ndim == 2:
        x = x.unsqueeze(0)

    x = x.to(torch.float32)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT

    x = x / (torch.linalg.vector_norm(x, dim=(-2, -1), keepdim=True) + eps)
    x = x.to(_gns_work_dtype(g, work_dtype))
    if max(x.shape[-2:]) > min(x.shape[-2:]):
        x = _gram_newton_schulz_core(x, steps, eps, bound_safety)
    else:
        x = _standard_newton_schulz_core(x, steps, eps, bound_safety)

    if transposed:
        x = x.mT
    x = x.to(original_dtype)
    if len(original_shape) == 2:
        return x.squeeze(0)
    return x.reshape(original_shape)


class ColNormLMO:
    __slots__ = ("eps", "transpose")

    def __init__(
        self, eps: float = 1e-12, transpose: bool = False
    ):
        self.eps = eps
        self.transpose = transpose

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        x = w.mT if self.transpose else w
        x = x.div_(
            torch.linalg.vector_norm(x, dim=0, keepdim=True).clamp_min_(self.eps)
        ).mul_(-math.sqrt(x.size(0)))
        return x.mT if self.transpose else x


class RowNormLMO:
    __slots__ = ("eps", "transpose")

    def __init__(
        self, eps: float = 1e-12, transpose: bool = False
    ):
        self.eps = eps
        self.transpose = transpose

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        x = w.mT if self.transpose else w
        x = x.div_(
            torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min_(self.eps)
        ).mul_(-1.0 / math.sqrt(x.size(1)))
        return x.mT if self.transpose else x


class SignLMO:
    __slots__ = ("transpose",)

    def __init__(self, transpose: bool = False):
        self.transpose = transpose

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        x = w.mT if self.transpose else w
        x = x.sign().mul_(-1.0 / x.size(1))
        return x.mT if self.transpose else x


class GramNewtonSchulzLMO:
    __slots__ = ("steps", "eps", "work_dtype", "input_like", "bound_safety")

    def __init__(
        self,
        steps: int = 5,
        eps: float = 1e-7,
        work_dtype: torch.dtype | None = None,
        input_like: bool = False,
        bound_safety: float = 1.05,
    ):
        self.steps = steps
        self.eps = eps
        self.work_dtype = work_dtype
        self.input_like = input_like
        self.bound_safety = bound_safety

    def _scale(self, x: torch.Tensor) -> float:
        scale = math.sqrt(x.size(-2) / x.size(-1))
        return max(1.0, scale) if self.input_like else scale

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 2:
            raise ValueError("GramNewtonSchulzLMO expects a 2D tensor")
        return gram_newton_schulz_uvt(
            v, self.steps, self.eps, self.work_dtype, self.bound_safety
        ).mul_(-self._scale(v))

    def batch(
        self, tensors: list[torch.Tensor], params: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor | None] = [None] * len(tensors)
        groups: dict[tuple, list[tuple[int, torch.Tensor]]] = {}

        for i, (x, _) in enumerate(zip(tensors, params, strict=True)):
            if x.ndim != 2:
                out[i] = self(x)
                continue
            key = (tuple(x.shape), x.dtype, x.device)
            groups.setdefault(key, []).append((i, x))

        for items in groups.values():
            y_batch = gram_newton_schulz_uvt(
                torch.stack([x for _, x in items]),
                self.steps,
                self.eps,
                self.work_dtype,
                self.bound_safety,
            )
            for j, (i, x) in enumerate(items):
                out[i] = y_batch[j].mul_(-self._scale(x))

        if any(x is None for x in out):
            raise RuntimeError("batched GramNewtonSchulzLMO missed an output")
        return out


SpectralLMO = GramNewtonSchulzLMO


@torch.no_grad()
def init_colnorm_(
    w: torch.Tensor,
    radius: float = 1.0,
    eps: float = 1e-12,
    transpose: bool = False,
) -> torch.Tensor:
    x = w.mT if transpose else w
    nn.init.normal_(x)
    x.div_(torch.linalg.vector_norm(x, dim=0, keepdim=True).clamp_min_(eps)).mul_(
        radius * math.sqrt(x.size(0))
    )
    return w


@torch.no_grad()
def init_rownorm_(
    w: torch.Tensor,
    radius: float = 1.0,
    eps: float = 1e-12,
    transpose: bool = False,
) -> torch.Tensor:
    x = w.mT if transpose else w
    nn.init.normal_(x)
    x.div_(torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min_(eps)).mul_(
        radius / math.sqrt(x.size(1))
    )
    return w


@torch.no_grad()
def init_sign_(
    w: torch.Tensor,
    radius: float = 1.0,
    transpose: bool = False,
) -> torch.Tensor:
    x = w.mT if transpose else w
    x.copy_(torch.randn_like(x).sign_())
    x.mul_(radius / x.size(1))
    return w


@torch.no_grad()
def init_spectral_(
    w: torch.Tensor, radius: float = 1.0, input_like: bool = False
) -> torch.Tensor:
    scale = math.sqrt(w.size(0) / w.size(1))
    if input_like:
        scale = max(1.0, scale)
    w_fp = w.data.double()
    nn.init.orthogonal_(w_fp)
    w_fp.mul_(radius * scale)
    w.data.copy_(w_fp.to(dtype=w.dtype))
    return w


@torch.no_grad()
def init_semiorthogonal_(
    w: torch.Tensor, radius: float = 1.0, input_like: bool = False
) -> torch.Tensor:
    return init_spectral_(w, radius=radius, input_like=input_like)


class ScionC(LionKCCWDPA):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        readout_mu: float = 1.0,
        memory_beta: float = 0.95,
        dir_fn=None,
        eps: float = 1e-12,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=(readout_mu, memory_beta),
            dir_fn=dir_fn,
            nesterov=True,
            eps=eps,
        )

