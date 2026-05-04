import math

import torch
import torch.nn as nn

from scionc.compile_env import ensure_compile_env
from scionc.ulmos.streaming_svd import HiddenSVDFilterULMO, StreamingSVDULMO

_GNSGroupKey = tuple[tuple[int, ...], torch.dtype, torch.device]
_GNSGroupItem = tuple[int, torch.Tensor, torch.Tensor, bool]

__all__ = [
    "ColNormULMO",
    "RowNormULMO",
    "GramNewtonSchulzULMO",
    "StreamingSVDULMO",
    "HiddenSVDFilterULMO",
    "SignULMO",
    "init_colnorm_",
    "init_rownorm_",
    "init_spectral_",
    "init_sign_",
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
_FP32_EPS = torch.finfo(torch.float32).eps
_MOMENT4_REFINE_STEPS = 8
_MOMENT4_FEAS_TOL = 0.25 * _FP32_EPS


def _gns_coeff(
    i: int, coeffs: tuple[tuple[float, float, float], ...]
) -> tuple[float, float, float]:
    return coeffs[i if i < len(coeffs) else -1]


def _gns_work_dtype(x: torch.Tensor, work_dtype: torch.dtype | None) -> torch.dtype:
    if work_dtype is not None:
        return work_dtype
    return torch.float16 if x.is_cuda else torch.float32


def _moment2_upper_beta(m2: torch.Tensor, n: int) -> torch.Tensor:
    v = (m2 - 1.0 / n).clamp_min(0.0)
    return 1.0 / n + torch.sqrt(((n - 1.0) / n) * v)


def _dot_roundoff_gamma(dot_dim: int, eps: float = _FP32_EPS) -> float:
    dim_eps = dot_dim * eps
    return dim_eps / (1.0 - dim_eps)


def _moment4_upper_beta(
    m2: torch.Tensor,
    m3: torch.Tensor,
    m4: torch.Tensor,
    n: int,
    eps: float,
    beta2: torch.Tensor,
    refine_steps: int = _MOMENT4_REFINE_STEPS,
    feas_tol: float = _MOMENT4_FEAS_TOL,
) -> torch.Tensor:
    if n <= 1:
        return beta2.clamp(eps, 1.0)

    n_f = float(n)
    lower = torch.sqrt(torch.sqrt(m4.clamp_min(eps)))
    upper = beta2

    d4 = 1.0 - n_f * m2
    d3 = 2.0 * (n_f * m3 - m2)
    d2 = 3.0 * m2.square() - 2.0 * m3 - n_f * m4
    d1 = 2.0 * (m4 - m2 * m3)
    d0 = (
        (n_f - 1.0) * (m2 * m4 - m3.square())
        - m2 * m2.square()
        + 2.0 * m2 * m3
        - m4
    )

    for _ in range(refine_steps):
        t = 0.5 * (lower + upper)
        t2 = t.square()

        p_t = (((d4 * t + d3) * t + d2) * t + d1) * t + d0
        e0 = (m2 - t2) * (m4 - t2.square()) - (m3 - t2 * t).square()

        feasible = torch.minimum(e0, p_t) >= -feas_tol
        lower = torch.where(feasible, t, lower)
        upper = torch.where(feasible, upper, t)

    return upper.clamp(eps, 1.0)


def _spectral_bounds_from_gram(
    gram: torch.Tensor,
    eps: float,
    safety: float,
    gram_square: torch.Tensor | None = None,
    dot_dim: int | None = None,
    fp32_eps: float = _FP32_EPS,
    refine_steps: int = _MOMENT4_REFINE_STEPS,
    feas_tol: float = _MOMENT4_FEAS_TOL,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = gram.size(-1)
    dot_dim = n if dot_dim is None else dot_dim

    trace_raw = gram.diagonal(dim1=-2, dim2=-1).sum(-1, dtype=torch.float32)
    active = trace_raw > eps
    trace = trace_raw.clamp_min(eps)
    trace_sq = trace.square()

    if gram_square is None:
        r2 = gram.float().square().sum(dim=(-2, -1), dtype=torch.float32)
        m2 = r2.div(trace_sq).clamp(1.0 / n, 1.0)
        lower_beta = m2
        upper_beta = _moment2_upper_beta(m2, n).clamp(eps, 1.0)
    else:
        gram_f = gram.float()
        gram_square_f = gram_square.float()
        r2 = gram_square_f.diagonal(dim1=-2, dim2=-1).sum(-1)
        r3 = (gram_f * gram_square_f).sum(dim=(-2, -1))
        r4 = gram_square_f.square().sum(dim=(-2, -1))

        m2 = r2.div(trace_sq).clamp(1.0 / n, 1.0)
        m3 = r3.div(trace_sq * trace).clamp(1.0 / (n * n), 1.0)
        m4 = r4.div(trace_sq.square()).clamp(1.0 / (n * n * n), 1.0)

        beta2 = _moment2_upper_beta(m2, n).clamp(eps, 1.0)
        beta4 = _moment4_upper_beta(
            m2, m3, m4, n, eps, beta2, refine_steps, feas_tol
        )
        upper_beta = torch.minimum(beta2, beta4.to(beta2.dtype)).clamp(eps, 1.0)
        lower_beta = torch.sqrt(torch.sqrt(m4.clamp_min(eps)))
        lower_beta = torch.minimum(lower_beta.to(upper_beta.dtype), upper_beta)

    lower = torch.where(
        active,
        (trace * lower_beta).clamp_min(0.0),
        torch.zeros_like(trace),
    )
    roundoff = _dot_roundoff_gamma(dot_dim, fp32_eps) * trace
    upper = (trace * upper_beta + roundoff).mul(safety).clamp_min(eps)
    upper = torch.where(active, upper, torch.ones_like(upper))
    return lower, upper


def _scale_gram_and_first_poly(
    x: torch.Tensor,
    gram: torch.Tensor,
    gram_square: torch.Tensor,
    eps: float,
    safety: float,
    fp32_eps: float = _FP32_EPS,
    refine_steps: int = _MOMENT4_REFINE_STEPS,
    feas_tol: float = _MOMENT4_FEAS_TOL,
    gns_coeffs: tuple[tuple[float, float, float], ...] = _GNS_COEFFS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, bound = _spectral_bounds_from_gram(
        gram,
        eps,
        safety,
        gram_square,
        x.size(-1),
        fp32_eps,
        refine_steps,
        feas_tol,
    )
    x_scale = torch.rsqrt(bound).reshape(-1, 1, 1).to(x.dtype)
    scale2 = x_scale.square()
    scale4 = scale2.square()
    gram = gram * scale2
    _, b0, c0 = gns_coeffs[0]
    first_poly = b0 * gram + c0 * (gram_square * scale4)
    return gram, first_poly, x_scale


ensure_compile_env()
_scale_gram_and_first_poly_cuda = torch.compile(
    _scale_gram_and_first_poly, fullgraph=True, dynamic=False
)


def _gram_newton_schulz_core(
    x: torch.Tensor,
    steps: int,
    eps: float,
    bound_safety: float,
    gns_coeffs: tuple[tuple[float, float, float], ...] = _GNS_COEFFS,
    gns_resets: frozenset[int] = _GNS_RESETS,
    fp32_eps: float = _FP32_EPS,
    refine_steps: int = _MOMENT4_REFINE_STEPS,
    feas_tol: float = _MOMENT4_FEAS_TOL,
) -> torch.Tensor:
    if steps <= 0:
        return x

    gram = torch.bmm(x, x.mT)
    gram_square = torch.bmm(gram, gram)
    scale_gram = (
        _scale_gram_and_first_poly_cuda if x.is_cuda else _scale_gram_and_first_poly
    )
    gram, z, x_scale = scale_gram(
        x,
        gram,
        gram_square,
        eps,
        bound_safety,
        fp32_eps,
        refine_steps,
        feas_tol,
        gns_coeffs,
    )
    a0, _, _ = gns_coeffs[0]
    eye = torch.eye(gram.size(-1), dtype=x.dtype, device=x.device).expand_as(gram)
    q = z + a0 * eye

    if steps > 1 and 1 not in gns_resets:
        rz = torch.baddbmm(gram, gram, z, beta=a0)
        gram = torch.baddbmm(rz, z, rz, beta=a0)

    for i in range(1, steps):
        a, b, c = _gns_coeff(i, gns_coeffs)
        reset = i in gns_resets
        if reset:
            if x_scale is not None:
                q = q * x_scale
                x_scale = None
            x = torch.bmm(q, x)
            gram = torch.bmm(x, x.mT)

        z = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        q = z + a * eye if reset else torch.baddbmm(q, q, z, beta=a)

        if i == steps - 1 or i + 1 in gns_resets:
            continue
        rz = torch.baddbmm(gram, gram, z, beta=a)
        gram = torch.baddbmm(rz, z, rz, beta=a)

    if x_scale is not None:
        q = q * x_scale
    return torch.bmm(q, x)


def _standard_newton_schulz_core(
    x: torch.Tensor,
    steps: int,
    eps: float,
    bound_safety: float,
    gns_coeffs: tuple[tuple[float, float, float], ...] = _GNS_COEFFS,
    fp32_eps: float = _FP32_EPS,
    refine_steps: int = _MOMENT4_REFINE_STEPS,
    feas_tol: float = _MOMENT4_FEAS_TOL,
) -> torch.Tensor:
    if steps <= 0:
        return x

    gram = torch.bmm(x, x.mT)
    gram_square = torch.bmm(gram, gram)
    scale_gram = (
        _scale_gram_and_first_poly_cuda if x.is_cuda else _scale_gram_and_first_poly
    )
    _, update, x_scale = scale_gram(
        x,
        gram,
        gram_square,
        eps,
        bound_safety,
        fp32_eps,
        refine_steps,
        feas_tol,
        gns_coeffs,
    )
    a0, _, _ = gns_coeffs[0]
    x = torch.baddbmm(x, update, x, beta=a0)
    x = x * x_scale

    for i in range(1, steps):
        a, b, c = _gns_coeff(i, gns_coeffs)
        gram = torch.bmm(x, x.mT)
        update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, update, x, beta=a)
    return x


def gram_newton_schulz_uvt(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    work_dtype: torch.dtype | None = None,
    bound_safety: float = 1.05,
    gns_coeffs: tuple[tuple[float, float, float], ...] = _GNS_COEFFS,
    gns_resets: frozenset[int] = _GNS_RESETS,
    fp32_eps: float = _FP32_EPS,
    refine_steps: int = _MOMENT4_REFINE_STEPS,
    feas_tol: float = _MOMENT4_FEAS_TOL,
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
        x = _gram_newton_schulz_core(
            x, steps, eps, bound_safety, gns_coeffs, gns_resets, fp32_eps, refine_steps, feas_tol
        )
    else:
        x = _standard_newton_schulz_core(
            x, steps, eps, bound_safety, gns_coeffs, fp32_eps, refine_steps, feas_tol
        )

    if transposed:
        x = x.mT
    x = x.to(original_dtype)
    if len(original_shape) == 2:
        return x.squeeze(0)
    return x.reshape(original_shape)


class ColNormULMO:
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


class RowNormULMO:
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


class SignULMO:
    __slots__ = ("transpose",)

    def __init__(self, transpose: bool = False):
        self.transpose = transpose

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        x = w.mT if self.transpose else w
        x = x.sign_().mul_(-1.0 / x.size(1))
        return x.mT if self.transpose else x


class GramNewtonSchulzULMO:
    __slots__ = (
        "steps",
        "eps",
        "work_dtype",
        "input_like",
        "bound_safety",
        "gns_coeffs",
        "gns_resets",
        "fp32_eps",
        "refine_steps",
        "feas_tol",
    )

    def __init__(
        self,
        steps: int = 5,
        eps: float = 1e-7,
        work_dtype: torch.dtype | None = None,
        input_like: bool = False,
        bound_safety: float = 1.05,
        gns_coeffs: tuple[tuple[float, float, float], ...] = _GNS_COEFFS,
        gns_resets: frozenset[int] = _GNS_RESETS,
        fp32_eps: float = _FP32_EPS,
        refine_steps: int = _MOMENT4_REFINE_STEPS,
        feas_tol: float = _MOMENT4_FEAS_TOL,
    ):
        self.steps = steps
        self.eps = eps
        self.work_dtype = work_dtype
        self.input_like = input_like
        self.bound_safety = bound_safety
        self.gns_coeffs = gns_coeffs
        self.gns_resets = gns_resets
        self.fp32_eps = fp32_eps
        self.refine_steps = refine_steps
        self.feas_tol = feas_tol

    def _scale(self, x: torch.Tensor) -> float:
        scale = math.sqrt(x.size(-2) / x.size(-1))
        return max(1.0, scale) if self.input_like else scale

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim != 2:
            raise ValueError("GramNewtonSchulzULMO expects a 2D tensor")
        return gram_newton_schulz_uvt(
            v,
            self.steps,
            self.eps,
            self.work_dtype,
            self.bound_safety,
            self.gns_coeffs,
            self.gns_resets,
            self.fp32_eps,
            self.refine_steps,
            self.feas_tol,
        ).mul_(-self._scale(v))

    def batch(
        self, tensors: list[torch.Tensor], params: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor | None] = [None] * len(tensors)
        groups: dict[_GNSGroupKey, list[_GNSGroupItem]] = {}

        for i, (x, _) in enumerate(zip(tensors, params, strict=True)):
            if x.ndim != 2:
                out[i] = self(x)
                continue
            transposed = x.size(0) > x.size(1)
            work = x.mT if transposed else x
            key = (tuple(work.shape), work.dtype, work.device)
            groups.setdefault(key, []).append((i, x, work, transposed))

        for items in groups.values():
            y_batch = gram_newton_schulz_uvt(
                torch.stack([work for _, _, work, _ in items]),
                self.steps,
                self.eps,
                self.work_dtype,
                self.bound_safety,
                self.gns_coeffs,
                self.gns_resets,
                self.fp32_eps,
                self.refine_steps,
                self.feas_tol,
            )
            scale = y_batch.new_tensor([-self._scale(x) for _, x, _, _ in items])
            y_batch.mul_(scale.reshape(-1, 1, 1))
            for j, (i, x, _, transposed) in enumerate(items):
                y = y_batch[j].mT if transposed else y_batch[j]
                out[i] = y

        if any(x is None for x in out):
            raise RuntimeError("batched GramNewtonSchulzULMO missed an output")
        return out


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

