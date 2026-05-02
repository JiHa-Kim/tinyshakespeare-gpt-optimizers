import math

import torch

_FilterEntry = tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]
_FilterGroupKey = tuple[tuple[int, ...], tuple[int, ...], torch.dtype, torch.device]
_SVDGroupKey = tuple[tuple[int, ...], torch.dtype, torch.device]
_SVDItem = tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, bool, float]


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
        "steps",
        "ridge",
        "refresh_interval",
        "refresh_threshold",
        "iteration",
        "eps",
        "work_dtype",
        "input_like",
        "states",
        "_param_key",
        "stats",
    )

    def __init__(
        self,
        steps: int = 1,
        ridge: float = 1e-3,
        refresh_interval: int = 25,
        refresh_threshold: float = 0.10,
        iteration: str = "scqr2",
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
        if iteration not in {"scqr2", "norm-power"}:
            raise ValueError(f"invalid iteration: {iteration}")

        self.steps = steps
        self.ridge = ridge
        self.refresh_interval = refresh_interval
        self.refresh_threshold = refresh_threshold
        self.iteration = iteration
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
        if gram.dtype not in {torch.float32, torch.float64}:
            gram = gram.float()
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
        if x.dtype != r.dtype:
            x = x.to(r.dtype)
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
            return torch.linalg.qr(b.float(), mode="reduced").Q.to(dtype=v.dtype)
        return self._qr(b.to(dtype=v.dtype)).to(dtype=v.dtype)

    def _v_step_norm_power(
        self, m: torch.Tensor, v: torch.Tensor, check_refresh: bool = False
    ) -> torch.Tensor:
        mv = m @ v
        mv_scaled, inv_scale = _normalize_columns(mv, self.eps)
        b = (m.mT @ mv) * inv_scale

        exact_qr = False
        if check_refresh:
            if self.refresh_threshold == 0.0:
                exact_qr = True
            else:
                quality = mv_scaled.mT @ mv_scaled
                n = quality.size(-1)
                offdiag_sq = quality.square().sum(dim=(-2, -1))
                offdiag_sq = offdiag_sq - quality.diagonal(
                    dim1=-2, dim2=-1
                ).square().sum(dim=-1)
                rms = torch.sqrt(offdiag_sq.clamp_min_(0.0) / max(n * (n - 1), 1))
                self.stats["quality_checks"] += int(rms.numel())
                exact_qr = bool((rms > self.refresh_threshold).any())

        if exact_qr:
            self.stats["refreshes"] += b.shape[0] if b.ndim == 3 else 1
            return torch.linalg.qr(b.float(), mode="reduced").Q.to(dtype=v.dtype)
        return self._qr(b.to(dtype=v.dtype)).to(dtype=v.dtype)

    def _v_step(
        self, m: torch.Tensor, v: torch.Tensor, check_refresh: bool = False
    ) -> torch.Tensor:
        if self.iteration == "norm-power":
            return self._v_step_norm_power(m, v, check_refresh)
        return self._v_step_scqr2(m, v, check_refresh)

    def _output_scale(
        self,
        items: list[_SVDItem],
        v_batch: torch.Tensor,
        mv: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
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
        groups: dict[_SVDGroupKey, list[_SVDItem]] = {}
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
            y_batch = (mv * mv_scale.unsqueeze(-2)) @ v_batch.mT

            for j, (i, x, p, m, transposed, scale) in enumerate(items):
                v = v_batch[j]
                self._store_basis_for(p, m, v)
                y = y_batch[j].mT if transposed else y_batch[j]
                out[i] = y.to(dtype=x.dtype).mul_(-scale)
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
        return out.to(dtype=x.dtype).mul_(-scale)


class HiddenSVDFilterLMO(StreamingSVDSpectralLMO):
    """
    Streaming-SVD hidden update with the closed-form diagonal filter for
    q(D)=tr(D A D^T), where A is the incoming activation covariance.
    """

    __slots__ = (
        "cov_accums",
        "cov_cache",
        "filter_ridge",
    )

    def __init__(
        self,
        *args,
        filter_ridge: float = 1e-3,
        **kwargs,
    ):
        if filter_ridge < 0.0:
            raise ValueError(f"invalid filter_ridge: {filter_ridge}")
        super().__init__(*args, **kwargs)
        self.cov_accums = {}
        self.cov_cache = {}
        self.filter_ridge = filter_ridge
        self.stats.update({"filtered": 0, "missing_covs": 0, "cov_updates": 0})

    def add_covariance(self, p: torch.Tensor, cov: torch.Tensor, count: int) -> None:
        state = self.cov_accums.get(id(p))
        if state is None:
            self.cov_accums[id(p)] = [cov, count]
        else:
            state[0] = state[0] + cov
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

    def _output_scale(
        self,
        items: list[_SVDItem],
        v_batch: torch.Tensor,
        mv: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        scales = list(sigma.reciprocal().unbind())
        groups: dict[_FilterGroupKey, list[_FilterEntry]] = {}
        for i, (_, _, p, _, transposed, _) in enumerate(items):
            cov = self._cov_for(p)
            if cov is None:
                self.stats["missing_covs"] += 1
                continue
            basis = (
                mv[i] / sigma[i].unsqueeze(0).clamp_min(self.eps)
                if transposed
                else v_batch[i]
            )
            key = (tuple(cov.shape), tuple(basis.shape), basis.dtype, basis.device)
            groups.setdefault(key, []).append((i, cov, basis, sigma[i]))

        for entries in groups.values():
            cov = torch.stack([x[1] for x in entries])
            basis = torch.stack([x[2] for x in entries])
            sig = torch.stack([x[3] for x in entries])
            coeff = self._coeff_batch(cov, basis, sig)
            self.stats["filtered"] += len(entries)
            for j, (i, _, _, _) in enumerate(entries):
                scales[i] = coeff[j] / sig[j].clamp_min(self.eps)

        return torch.stack(scales)

    def _coeff_batch(
        self, cov: torch.Tensor, basis: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        cov = cov.to(device=basis.device, dtype=basis.dtype)
        denom = (cov @ basis).mul(basis).sum(dim=-2).clamp_min(self.eps)
        return self._coeff_from_denom(denom, sigma)

    def batch(
        self, tensors: list[torch.Tensor], params: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        try:
            return super().batch(tensors, params)
        finally:
            self.cov_accums.clear()

