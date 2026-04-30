import math

import torch
import torch.nn as nn

from lionk_ccwd import LionKCCWDPA, lionk_S

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
    "scion_transfer_lr",
    "Scion",
    "ScionC",
]


_GNS_COEFFS = (
    (
        8.28721201814563 / 1.05,
        -23.595886519098837 / (1.05**3),
        17.300387312530933 / (1.05**5),
    ),
    (
        4.107059111542203 / 1.05,
        -2.9478499167379106 / (1.05**3),
        0.5448431082926601 / (1.05**5),
    ),
    (
        3.9486908534822946 / 1.05,
        -2.908902115962949 / (1.05**3),
        0.5518191394370137 / (1.05**5),
    ),
    (
        3.3184196573706015 / 1.05,
        -2.488488024314874 / (1.05**3),
        0.51004894012372 / (1.05**5),
    ),
    (
        2.300652019954817 / 1.05,
        -1.6689039845747493 / (1.05**3),
        0.4188073119525673 / (1.05**5),
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
    __slots__ = ("radius", "eps", "transpose")

    def __init__(
        self, radius: float = 1.0, eps: float = 1e-12, transpose: bool = False
    ):
        self.radius = radius
        self.eps = eps
        self.transpose = transpose

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        x = w.mT if self.transpose else w
        x = x.div_(
            torch.linalg.vector_norm(x, dim=0, keepdim=True).clamp_min_(self.eps)
        ).mul_(-self.radius * math.sqrt(x.size(0)))
        return x.mT if self.transpose else x


class RowNormLMO:
    __slots__ = ("radius", "eps", "transpose")

    def __init__(
        self, radius: float = 10.0, eps: float = 1e-12, transpose: bool = False
    ):
        self.radius = radius
        self.eps = eps
        self.transpose = transpose

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        x = w.mT if self.transpose else w
        x = x.div_(
            torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min_(self.eps)
        ).mul_(-self.radius / math.sqrt(x.size(1)))
        return x.mT if self.transpose else x


class SignLMO:
    __slots__ = ("radius", "transpose")

    def __init__(self, radius: float = 10.0, transpose: bool = False):
        self.radius = radius
        self.transpose = transpose

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        x = w.mT if self.transpose else w
        x = x.sign().mul_(-self.radius / x.size(1))
        return x.mT if self.transpose else x


class GramNewtonSchulzLMO:
    __slots__ = ("radius", "steps", "eps", "work_dtype", "input_like", "bound_safety")

    def __init__(
        self,
        radius: float = 3.0,
        steps: int = 5,
        eps: float = 1e-7,
        work_dtype: torch.dtype | None = None,
        input_like: bool = False,
        bound_safety: float = 1.05,
    ):
        self.radius = radius
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
        ).mul_(-self.radius * self._scale(v))

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
                out[i] = y_batch[j].mul_(-self.radius * self._scale(x))

        if any(x is None for x in out):
            raise RuntimeError("batched GramNewtonSchulzLMO missed an output")
        return out


SpectralLMO = GramNewtonSchulzLMO


def _column_inverse_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    norms = torch.linalg.vector_norm(x, dim=-2, keepdim=True).clamp_min(eps)
    return norms.reciprocal()


def _normalize_columns(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_scale = _column_inverse_scale(x, eps)
    return x * inv_scale, inv_scale


class StreamingSVDSpectralLMO:
    """
    Spectral LMO based on one or more streaming power-iteration steps.

    The cached V basis is stored per parameter via `set_param`, which is called
    by LionKCCWDPA when the LMO exposes that hook.
    """

    __slots__ = (
        "radius",
        "steps",
        "ridge",
        "refresh_interval",
        "refresh_threshold",
        "eps",
        "work_dtype",
        "input_like",
        "states",
        "_param_key",
        "stats",
    )

    def __init__(
        self,
        radius: float = 3.0,
        steps: int = 1,
        ridge: float = 1e-3,
        refresh_interval: int = 25,
        refresh_threshold: float = 0.10,
        eps: float = 1e-12,
        work_dtype: torch.dtype | None = torch.float32,
        input_like: bool = False,
    ):
        if steps <= 0:
            raise ValueError(f"invalid steps: {steps}")
        if ridge < 0.0:
            raise ValueError(f"invalid ridge: {ridge}")
        if refresh_interval < 0:
            raise ValueError(f"invalid refresh_interval: {refresh_interval}")
        if refresh_threshold < 0.0:
            raise ValueError(f"invalid refresh_threshold: {refresh_threshold}")

        self.radius = radius
        self.steps = steps
        self.ridge = ridge
        self.refresh_interval = refresh_interval
        self.refresh_threshold = refresh_threshold
        self.eps = eps
        self.work_dtype = work_dtype
        self.input_like = input_like
        self.states = {}
        self._param_key = None
        self.stats = {"calls": 0, "steps": 0, "refreshes": 0, "quality_checks": 0}

    def set_param(self, p: torch.Tensor) -> None:
        self._param_key = id(p)

    def _state_key(self, x: torch.Tensor) -> tuple:
        base = self._param_key
        if base is None:
            base = ("shape", tuple(x.shape))
        return (base, tuple(x.shape), x.dtype, x.device)

    def _resolve_work_dtype(self, x: torch.Tensor) -> torch.dtype:
        if self.work_dtype is not None:
            return self.work_dtype
        if x.dtype in {torch.float16, torch.bfloat16}:
            return torch.float32
        return x.dtype

    def _ridge_scale(self, gram: torch.Tensor) -> torch.Tensor:
        scale = gram[..., 0, 0].abs()
        return torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(
            self.eps
        )

    def _cholesky_fast(self, gram: torch.Tensor, ridge: float) -> torch.Tensor:
        scale = self._ridge_scale(gram)
        shifted = (gram + gram.mT).mul(0.5)
        diag = shifted.diagonal(dim1=-2, dim2=-1)
        shift = ridge * scale
        if shift.ndim:
            shift = shift.unsqueeze(-1)
        diag.add_(shift)
        r, _ = torch.linalg.cholesky_ex(shifted, upper=True, check_errors=False)
        return r

    def _solve_right(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(r, x, upper=True, left=False)

    def _scqr_once_fast(self, x: torch.Tensor, ridge: float) -> torch.Tensor:
        r = self._cholesky_fast(x.mT @ x, ridge)
        return self._solve_right(x, r)

    def _qr(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled, _ = _normalize_columns(x, self.eps)
        return self._scqr_once_fast(x_scaled, self.ridge)

    def _v_step_scqr2(
        self, m: torch.Tensor, v: torch.Tensor, check_refresh: bool = False
    ) -> torch.Tensor:
        mv = m @ v
        mv_scaled, inv_scale = _normalize_columns(mv, self.eps)

        # Direct Gram from M @ V preserves positive semidefiniteness better than
        # the algebraically equivalent V.T @ (M.T @ M @ V) in finite precision.
        a_scaled = (m.mT @ mv) * inv_scale
        gram1 = mv_scaled.mT @ mv_scaled
        exact_qr = False
        if check_refresh:
            if self.refresh_threshold == 0.0:
                exact_qr = True
            else:
                quality = gram1.detach().clone()
                quality.diagonal(dim1=-2, dim2=-1).zero_()
                n = quality.size(-1)
                rms = torch.sqrt(
                    quality.square().sum(dim=(-2, -1)) / max(n * (n - 1), 1)
                )
                self.stats["quality_checks"] += int(rms.numel())
                exact_qr = bool((rms > self.refresh_threshold).any())
        r1 = self._cholesky_fast(gram1, self.ridge)
        b = self._solve_right(a_scaled, r1)
        if exact_qr:
            self.stats["refreshes"] += b.shape[0] if b.ndim == 3 else 1
            return torch.linalg.qr(b, mode="reduced").Q
        return self._qr(b)

    def _v_step(
        self, m: torch.Tensor, v: torch.Tensor, check_refresh: bool = False
    ) -> torch.Tensor:
        return self._v_step_scqr2(m, v, check_refresh)

    def _filter_coefficients(
        self,
        items: list[tuple],
        v_batch: torch.Tensor,
        u_batch: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor | None:
        return None

    def _output_scale(
        self,
        items: list[tuple],
        v_batch: torch.Tensor,
        mv: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor | None:
        return sigma.reciprocal()

    def _basis_for(self, p: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        key = (id(p), tuple(m.shape), m.dtype, m.device)
        v = self.states.get(key)
        if v is None or v.shape != (m.size(-1), m.size(-1)):
            v = torch.eye(m.size(-1), dtype=m.dtype, device=m.device)
        return v

    def _store_basis_for(self, p: torch.Tensor, m: torch.Tensor, v: torch.Tensor) -> None:
        key = (id(p), tuple(m.shape), m.dtype, m.device)
        self.states[key] = v.detach()

    def batch(
        self, tensors: list[torch.Tensor], params: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor | None] = [None] * len(tensors)
        groups: dict[tuple, list[tuple]] = {}
        self.stats["steps"] += 1
        check_refresh = (
            self.refresh_interval > 0
            and self.stats["steps"] % self.refresh_interval == 0
        )

        for i, (x, p) in enumerate(zip(tensors, params, strict=True)):
            if x.ndim != 2:
                out[i] = self(x)
                continue
            work_dtype = self._resolve_work_dtype(x)
            m = x.to(work_dtype)
            transposed = m.size(0) < m.size(1)
            if transposed:
                m = m.mT
            scale = math.sqrt(x.size(0) / x.size(1))
            if self.input_like:
                scale = max(1.0, scale)
            key = (tuple(m.shape), m.dtype, m.device)
            groups.setdefault(key, []).append((i, x, p, m, transposed, scale))

        for items in groups.values():
            m_batch = torch.stack([item[3] for item in items])
            v_batch = torch.stack([self._basis_for(item[2], item[3]) for item in items])

            for _ in range(self.steps):
                v_batch = self._v_step(m_batch, v_batch, check_refresh)

            mv = m_batch @ v_batch
            sigma = torch.linalg.vector_norm(mv, dim=-2).clamp_min(self.eps)
            mv_scale = self._output_scale(items, v_batch, mv, sigma)
            if mv_scale is None:
                u = mv / sigma.unsqueeze(-2)
                coeff = self._filter_coefficients(items, v_batch, u, sigma)
                if coeff is not None:
                    u = u * coeff.unsqueeze(-2)
                y_batch = u @ v_batch.mT
            else:
                y_batch = (mv * mv_scale.unsqueeze(-2)) @ v_batch.mT

            for j, (i, x, p, m, transposed, scale) in enumerate(items):
                v = v_batch[j]
                self._store_basis_for(p, m, v)
                y = y_batch[j].mT if transposed else y_batch[j]
                out[i] = y.to(dtype=x.dtype).mul_(-self.radius * scale)
                self.stats["calls"] += 1

        if any(x is None for x in out):
            raise RuntimeError("batched StreamingSVDSpectralLMO missed an output")
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("StreamingSVDSpectralLMO expects a 2D tensor")

        scale = math.sqrt(x.size(0) / x.size(1))
        if self.input_like:
            scale = max(1.0, scale)

        work_dtype = self._resolve_work_dtype(x)
        m = x.to(work_dtype)
        transposed = m.size(0) < m.size(1)
        if transposed:
            m = m.mT
        self.stats["steps"] += 1
        check_refresh = (
            self.refresh_interval > 0
            and self.stats["steps"] % self.refresh_interval == 0
        )

        key = self._state_key(m)
        v = self.states.get(key)
        if v is None or v.shape != (m.size(1), m.size(1)):
            v = torch.eye(m.size(1), dtype=m.dtype, device=m.device)

        for _ in range(self.steps):
            v = self._v_step(m, v, check_refresh)
        self.states[key] = v.detach()

        mv = m @ v
        sigma = torch.linalg.vector_norm(mv, dim=0).clamp_min(self.eps)
        out = (mv * sigma.reciprocal().unsqueeze(0)) @ v.mT
        if transposed:
            out = out.mT
        self.stats["calls"] += 1
        return out.to(dtype=x.dtype).mul_(-self.radius * scale)


class HiddenSVDFilterLMO(StreamingSVDSpectralLMO):
    """
    Streaming-SVD hidden update with the closed-form diagonal filter for
    q(D)=tr(D A D^T), where A is the incoming activation covariance.
    """

    __slots__ = (
        "cov_accums",
        "cov_cache",
        "filter_ridge",
        "cov_interval",
        "filter_metric",
    )

    def __init__(
        self,
        *args,
        filter_ridge: float = 1e-3,
        cov_interval: int = 1,
        filter_metric: str = "grad-sigma",
        **kwargs,
    ):
        if filter_ridge < 0.0:
            raise ValueError(f"invalid filter_ridge: {filter_ridge}")
        if cov_interval <= 0:
            raise ValueError(f"invalid cov_interval: {cov_interval}")
        if filter_metric not in {"full", "grad-sigma"}:
            raise ValueError(f"invalid filter_metric: {filter_metric}")
        super().__init__(*args, **kwargs)
        self.cov_accums = {}
        self.cov_cache = {}
        self.filter_ridge = filter_ridge
        self.cov_interval = cov_interval
        self.filter_metric = filter_metric
        self.stats.update({"filtered": 0, "missing_covs": 0, "cov_updates": 0})

    def collect_covariance(self) -> bool:
        return self.stats["steps"] % self.cov_interval == 0

    def add_covariance(self, p: torch.Tensor, cov: torch.Tensor, count: int) -> None:
        state = self.cov_accums.get(id(p))
        if state is None:
            self.cov_accums[id(p)] = [cov, count]
        else:
            state[0].add_(cov)
            state[1] += count

    def _cov_for(self, p: torch.Tensor) -> torch.Tensor | None:
        key = id(p)
        state = self.cov_accums.get(key)
        if state is not None:
            cov_sum, count = state
            if count > 0:
                self.cov_cache[key] = (cov_sum / count).detach()
                self.stats["cov_updates"] += 1
        return self.cov_cache.get(key)

    def _coeff_from_denom(
        self, denom: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        denom = denom.clamp_min(self.eps)
        scale = denom.mean(dim=-1, keepdim=True).abs().clamp_min(self.eps)
        denom = denom + self.filter_ridge * scale

        raw = torch.nan_to_num(sigma / denom, nan=0.0, posinf=0.0, neginf=0.0)
        budget = denom.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        q_norm = denom.mul(raw.square()).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return raw * torch.sqrt(budget / q_norm)

    def _grad_sigma_mv_scale(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma2 = sigma.square()
        denom = sigma2.clamp_min(self.eps)
        scale = denom.mean(dim=-1, keepdim=True).abs().clamp_min(self.eps)
        denom = denom + self.filter_ridge * scale
        budget = denom.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        q_norm = sigma2.div(denom).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        factor = torch.sqrt(budget / q_norm).div(denom)
        return torch.nan_to_num(factor, nan=0.0, posinf=0.0, neginf=0.0)

    def _output_scale(
        self,
        items: list[tuple],
        v_batch: torch.Tensor,
        mv: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.filter_metric == "grad-sigma":
            self.stats["filtered"] += len(items)
            return self._grad_sigma_mv_scale(sigma)
        return None

    def _coeff_batch(
        self, cov: torch.Tensor, basis: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        cov = cov.to(device=basis.device, dtype=basis.dtype)
        denom = (cov @ basis).mul(basis).sum(dim=-2).clamp_min(self.eps)
        return self._coeff_from_denom(denom, sigma)

    def _filter_coefficients(
        self,
        items: list[tuple],
        v_batch: torch.Tensor,
        u_batch: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor | None:
        coeffs: list[torch.Tensor | None] = [None] * len(items)
        groups: dict[tuple, list[tuple]] = {}

        if self.filter_metric == "grad-sigma":
            self.stats["filtered"] += len(items)
            return self._coeff_from_denom(sigma.square(), sigma)

        for i, item in enumerate(items):
            basis = u_batch[i] if item[4] else v_batch[i]
            cov = self._cov_for(item[2])
            if cov is None:
                self.stats["missing_covs"] += 1
                coeffs[i] = torch.ones_like(sigma[i])
                continue
            key = (tuple(cov.shape), tuple(basis.shape), basis.dtype, basis.device)
            groups.setdefault(key, []).append((i, cov, basis, sigma[i]))

        for entries in groups.values():
            cov = torch.stack([x[1] for x in entries])
            basis = torch.stack([x[2] for x in entries])
            sig = torch.stack([x[3] for x in entries])
            coeff = self._coeff_batch(cov, basis, sig)
            self.stats["filtered"] += len(entries)
            for j, (i, _, _, _) in enumerate(entries):
                coeffs[i] = coeff[j]

        if any(x is None for x in coeffs):
            raise RuntimeError("HiddenSVDFilterLMO missed a coefficient")
        return torch.stack(coeffs)

    def batch(
        self, tensors: list[torch.Tensor], params: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        try:
            return super().batch(tensors, params)
        finally:
            self.cov_accums.clear()


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


def scion_transfer_lr(lr: float, mT: float = 1.0, mL: float = 1.0, alpha: float = 0.5):
    if lr <= 0.0:
        raise ValueError(f"invalid lr: {lr}")
    if mT <= 0.0:
        raise ValueError(f"invalid mT: {mT}")
    if mL <= 0.0:
        raise ValueError(f"invalid mL: {mL}")
    token_factor = mT**-0.5
    depth_factor = mL ** (alpha - 1.0)
    return {
        "embed": lr * token_factor,
        "hidden": lr * token_factor * depth_factor,
        "out": lr * token_factor,
    }


class Scion(LionKCCWDPA):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta2: float = 0.95,
        dir_fn=None,
        phi: float = 0.0,
        eta: float | None = None,
        theta2: float | None = None,
        cu2: float = 1.0,
        S: float | None = None,
        q: float = 1.0,
        cwd: bool = False,
        nesterov: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=(1.0, beta2),
            dir_fn=dir_fn,
            phi=phi,
            eta=eta,
            theta2=theta2,
            cu2=cu2,
            S=1.0 if S is None else S,
            q=q,
            cwd=cwd,
            nesterov=nesterov,
            eps=eps,
        )


class ScionC(LionKCCWDPA):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        beta2: float = 0.95,
        dir_fn=None,
        phi: float = 0.0,
        eta: float | None = None,
        theta2: float | None = None,
        cu2: float = 1.0,
        S: float | None = None,
        q: float = 1.0,
        cwd: bool = False,
        nesterov: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__(
            params=params,
            lr=lr,
            betas=(1.0, beta2),
            dir_fn=dir_fn,
            phi=phi,
            eta=eta,
            theta2=theta2,
            cu2=cu2,
            S=lionk_S(1.0, beta2, nesterov=nesterov) if S is None else S,
            q=q,
            cwd=cwd,
            nesterov=nesterov,
            eps=eps,
        )
