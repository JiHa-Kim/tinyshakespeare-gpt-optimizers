import math

import torch
import torch.nn as nn

from scionc.ulmos.streaming_svd import HiddenSVDFilterULMO, StreamingSVDULMO

_GNSGroupKey = tuple[tuple[int, ...], torch.dtype, torch.device]
_GNSGroupItem = tuple[int, torch.Tensor, torch.Tensor, bool]

__all__ = [
    "ColNormULMO",
    "RowNormULMO",
    "GramNewtonSchulzULMO",
    "SpectralULMO",
    "StreamingSVDULMO",
    "HiddenSVDFilterULMO",
    "SignULMO",
    "init_colnorm_",
    "init_rownorm_",
    "init_spectral_",
    "init_sign_",
    "init_semiorthogonal_",
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
_GNS_A0, _GNS_B0, _GNS_C0 = _GNS_COEFFS[0]
_FP32_EPS = torch.finfo(torch.float32).eps
_MOMENT4_REFINE_STEPS = 8
_MOMENT4_FEAS_TOL = 0.25 * _FP32_EPS
_MOMENT4_BETA_PAD = 2048.0 * _FP32_EPS


def _gns_coeff(i: int) -> tuple[float, float, float]:
    return _GNS_COEFFS[i if i < len(_GNS_COEFFS) else -1]


def _gns_work_dtype(x: torch.Tensor, work_dtype: torch.dtype | None) -> torch.dtype:
    if work_dtype is not None:
        return work_dtype
    return torch.float16 if x.is_cuda else torch.float32


def _moment2_beta_from_m2(m2: torch.Tensor, n: int) -> torch.Tensor:
    v = (m2 - 1.0 / n).clamp_min(0.0)
    return 1.0 / n + torch.sqrt(((n - 1.0) / n) * v)


def _moment4_coupled_feasible(
    t: torch.Tensor,
    m2: torch.Tensor,
    m3: torch.Tensor,
    m4: torch.Tensor,
    n: int,
    tol: float,
) -> torch.Tensor:
    n_float = float(n)
    t2 = t.square()
    t3 = t2 * t
    t4 = t2.square()
    m2_sq = m2.square()
    m3_sq = m3.square()
    m2m3 = m2 * m3
    m2m4 = m2 * m4
    moment_gap = m2m4 - m3_sq

    e0 = (m2 - t2) * (m4 - t4) - (m3 - t3).square()
    d0 = (
        (1.0 - n_float * m2) * t4
        + 2.0 * (n_float * m3 - m2) * t3
        + (3.0 * m2_sq - 2.0 * m3 - n_float * m4) * t2
        + 2.0 * (m4 - m2m3) * t
        + (n_float - 1.0) * moment_gap
        - m2 * m2_sq
        + 2.0 * m2m3
        - m4
    )
    return torch.minimum(e0, d0) >= -tol


def _moment4_support_lower(
    m2: torch.Tensor, m3: torch.Tensor, m4: torch.Tensor, eps: float
) -> torch.Tensor:
    q_a = (m3 - m2.square()).clamp_min(0.0)
    q_b = m2 * m3 - m4
    q_c = m2 * m4 - m3.square()
    q_disc = (q_b.square() - 4.0 * q_a * q_c).clamp_min(0.0)
    q_root = (-q_b + torch.sqrt(q_disc)) / (2.0 * q_a).clamp_min(eps)
    linear_root = (-q_c / q_b.clamp_max(-eps)).clamp_min(0.0)
    q_root = torch.where(q_a > eps, q_root, linear_root)
    return torch.maximum(torch.maximum(m2, m4 / m3.clamp_min(eps)), q_root)


def _moment4_upper_beta(
    m2: torch.Tensor, m3: torch.Tensor, m4: torch.Tensor, n: int, eps: float
) -> torch.Tensor:
    beta2 = _moment2_beta_from_m2(m2, n).clamp(eps, 1.0)
    if n <= 1:
        return beta2

    beta_m4 = torch.sqrt(torch.sqrt(m4.clamp_min(eps)))
    c_rad = ((n - 1.0) * (n * m4 - m2.square())).clamp_min(0.0)
    beta_c = torch.sqrt(((m2 + torch.sqrt(c_rad)) / n).clamp_min(eps))
    hi = torch.minimum(beta2, torch.minimum(beta_m4, beta_c)).clamp(eps, 1.0)
    lo = torch.minimum(_moment4_support_lower(m2, m3, m4, eps), hi)

    # The bracket enforces the one-dimensional PSD minors; only E and det(M0)
    # can still cut the interval.
    for _ in range(_MOMENT4_REFINE_STEPS):
        mid = 0.5 * (lo + hi)
        mid_feasible = _moment4_coupled_feasible(
            mid, m2, m3, m4, n, _MOMENT4_FEAS_TOL
        )
        lo = torch.where(mid_feasible, mid, lo)
        hi = torch.where(mid_feasible, hi, mid)

    return hi.clamp(eps, 1.0)


def _spectral_bound_from_gram(
    gram: torch.Tensor,
    eps: float,
    safety: float,
    gram_square: torch.Tensor | None = None,
) -> torch.Tensor:
    n = gram.size(-1)
    t1_raw = gram.diagonal(dim1=-2, dim2=-1).sum(-1, dtype=torch.float32)
    active = t1_raw > eps
    t1 = t1_raw.clamp_min(eps)
    if gram_square is None:
        r2 = gram.float().square().sum(dim=(-2, -1), dtype=torch.float32)
    else:
        r2 = gram_square.diagonal(dim1=-2, dim2=-1).sum(-1, dtype=torch.float32)

    m2 = r2.div(t1.square()).clamp(1.0 / n, 1.0)
    beta = _moment2_beta_from_m2(m2, n).clamp(eps, 1.0)

    if gram_square is not None and n > 1:
        t1_2 = t1.square()
        r3 = (gram.float() * gram_square.float()).sum(
            dim=(-2, -1), dtype=torch.float32
        )
        r4 = gram_square.float().square().sum(dim=(-2, -1), dtype=torch.float32)
        m3 = r3.div(t1_2 * t1).clamp(1.0 / (n * n), 1.0)
        m4 = r4.div(t1_2.square()).clamp(1.0 / (n * n * n), 1.0)
        beta4 = _moment4_upper_beta(m2, m3, m4, n, eps)
        beta = torch.minimum(beta, beta4.to(beta.dtype)).clamp(eps, 1.0)

    beta = (beta + _MOMENT4_BETA_PAD).clamp(eps, 1.0)
    bound = (t1 * beta + eps * t1).mul(safety).clamp_min(eps)
    return torch.where(active, bound, torch.ones_like(bound))


def _scale_gram_and_first_poly_eager(
    x: torch.Tensor,
    gram: torch.Tensor,
    gram_square: torch.Tensor,
    eps: float,
    safety: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bound = _spectral_bound_from_gram(gram, eps, safety, gram_square)
    x_scale = torch.rsqrt(bound).reshape(-1, 1, 1).to(x.dtype)
    scale2 = x_scale.square()
    scale4 = scale2.square()
    gram = gram * scale2
    first_poly = _GNS_B0 * gram + _GNS_C0 * (gram_square * scale4)
    return gram, first_poly, x_scale


_scale_gram_and_first_poly_compiled = torch.compile(
    _scale_gram_and_first_poly_eager, fullgraph=True, dynamic=False
)


def _scale_gram_and_first_poly(
    x: torch.Tensor,
    gram: torch.Tensor,
    gram_square: torch.Tensor,
    eps: float,
    safety: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.is_cuda:
        return _scale_gram_and_first_poly_compiled(x, gram, gram_square, eps, safety)
    return _scale_gram_and_first_poly_eager(x, gram, gram_square, eps, safety)


def _gram_newton_schulz_core(
    x: torch.Tensor, steps: int, eps: float, bound_safety: float
) -> torch.Tensor:
    if steps <= 0:
        return x

    gram = torch.bmm(x, x.mT)
    gram_square = torch.bmm(gram, gram)
    gram, z, x_scale = _scale_gram_and_first_poly(
        x, gram, gram_square, eps, bound_safety
    )
    eye = torch.eye(gram.size(-1), dtype=x.dtype, device=x.device).expand_as(gram)
    q = z + _GNS_A0 * eye

    if steps > 1 and 1 not in _GNS_RESETS:
        rz = torch.baddbmm(gram, gram, z, beta=_GNS_A0)
        gram = torch.baddbmm(rz, z, rz, beta=_GNS_A0)

    for i in range(1, steps):
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
    if steps <= 0:
        return x

    gram = torch.bmm(x, x.mT)
    gram_square = torch.bmm(gram, gram)
    gram, update, x_scale = _scale_gram_and_first_poly(
        x, gram, gram_square, eps, bound_safety
    )
    x = torch.baddbmm(x, update, x, beta=_GNS_A0)
    if x_scale is not None:
        x = x * x_scale

    for i in range(1, steps):
        a, b, c = _gns_coeff(i)
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
            raise ValueError("GramNewtonSchulzULMO expects a 2D tensor")
        return gram_newton_schulz_uvt(
            v, self.steps, self.eps, self.work_dtype, self.bound_safety
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
            )
            for j, (i, x, _, transposed) in enumerate(items):
                y = y_batch[j].mT if transposed else y_batch[j]
                out[i] = y.mul_(-self._scale(x))

        if any(x is None for x in out):
            raise RuntimeError("batched GramNewtonSchulzULMO missed an output")
        return out


SpectralULMO = GramNewtonSchulzULMO


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

