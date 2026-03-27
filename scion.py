import math

import torch
import torch.nn as nn

from lionk_ccwd import LionKCCWDPA, lionk_S

__all__ = [
    'ColNormLMO',
    'RowNormLMO',
    'SpectralLMO',
    'init_colnorm_',
    'init_rownorm_',
    'init_spectral_',
    'init_semiorthogonal_',
    'scion_transfer_lr',
    'Scion',
    'ScionC',
]


_PE = (
    (8.28721201814563 / 1.01, -23.595886519098837 / (1.01**3), 17.300387312530933 / (1.01**5)),
    (4.107059111542203 / 1.01, -2.9478499167379106 / (1.01**3), 0.5448431082926601 / (1.01**5)),
    (3.9486908534822946 / 1.01, -2.908902115962949 / (1.01**3), 0.5518191394370137 / (1.01**5)),
    (3.3184196573706015 / 1.01, -2.488488024314874 / (1.01**3), 0.51004894012372 / (1.01**5)),
    (2.300652019954817 / 1.01, -1.6689039845747493 / (1.01**3), 0.4188073119525673 / (1.01**5)),
    (1.891301407787398 / 1.01, -1.2679958271945868 / (1.01**3), 0.37680408948524835 / (1.01**5)),
    (1.8750014808534479 / 1.01, -1.2500016453999487 / (1.01**3), 0.3750001645474248 / (1.01**5)),
    (1.875, -1.25, 0.375),
)


def polar_express_uvt(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    work_dtype: torch.dtype | None = None,
    workspace: dict | None = None,
) -> torch.Tensor:
    if g.ndim != 2:
        raise ValueError('polar_express_uvt expects a 2D tensor')

    if work_dtype is None:
        work_dtype = torch.bfloat16 if g.is_cuda else g.dtype
    x = g.to(work_dtype)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.mT

    x = x / (torch.linalg.vector_norm(x) * 1.01 + eps)
    n = len(_PE)
    buffers = None if workspace is None else workspace.setdefault((tuple(x.shape), x.dtype, x.device), {})
    if buffers is not None:
        rows, cols = x.shape
        gram_shape = (rows, rows)
        if buffers.get('A') is None or tuple(buffers['A'].shape) != gram_shape:
            buffers['A'] = torch.empty(gram_shape, dtype=x.dtype, device=x.device)
        if buffers.get('AX') is None or tuple(buffers['AX'].shape) != (rows, cols):
            buffers['AX'] = torch.empty((rows, cols), dtype=x.dtype, device=x.device)
        if buffers.get('AAX') is None or tuple(buffers['AAX'].shape) != (rows, cols):
            buffers['AAX'] = torch.empty((rows, cols), dtype=x.dtype, device=x.device)

    for i in range(steps):
        a, b, c = _PE[i if i < n else n - 1]
        if buffers is None:
            A = x @ x.mT
            AX = A @ x
            AAX = A @ AX
        else:
            A = buffers['A']
            AX = buffers['AX']
            AAX = buffers['AAX']
            torch.mm(x, x.mT, out=A)
            torch.mm(A, x, out=AX)
            torch.mm(A, AX, out=AAX)
        x.mul_(a).add_(AX, alpha=b).add_(AAX, alpha=c)

    return (x.mT if transposed else x).to(g.dtype)


class ColNormLMO:
    __slots__ = ('radius', 'eps', 'transpose')

    def __init__(self, radius: float = 1.0, eps: float = 1e-12, transpose: bool = False):
        self.radius = radius
        self.eps = eps
        self.transpose = transpose

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        x = w.mT if self.transpose else w
        x = x.div_(torch.linalg.vector_norm(x, dim=0, keepdim=True).clamp_min_(self.eps)).mul_(
            -self.radius * math.sqrt(x.size(0))
        )
        return x.mT if self.transpose else x


class RowNormLMO:
    __slots__ = ('radius', 'eps', 'transpose')

    def __init__(self, radius: float = 10.0, eps: float = 1e-12, transpose: bool = False):
        self.radius = radius
        self.eps = eps
        self.transpose = transpose

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        x = w.mT if self.transpose else w
        x = x.div_(torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min_(self.eps)).mul_(
            -self.radius / math.sqrt(x.size(1))
        )
        return x.mT if self.transpose else x


class SpectralLMO:
    __slots__ = ('radius', 'steps', 'eps', 'work_dtype', 'input_like', 'workspace')

    def __init__(
        self,
        radius: float = 3.0,
        steps: int = 5,
        eps: float = 1e-7,
        work_dtype: torch.dtype | None = None,
        input_like: bool = False,
    ):
        self.radius = radius
        self.steps = steps
        self.eps = eps
        self.work_dtype = work_dtype
        self.input_like = input_like
        self.workspace = {}

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(v.size(0) / v.size(1))
        if self.input_like:
            scale = max(1.0, scale)
        return polar_express_uvt(v, self.steps, self.eps, self.work_dtype, self.workspace).mul_(-self.radius * scale)


@torch.no_grad()
def init_colnorm_(
    w: torch.Tensor,
    radius: float = 1.0,
    eps: float = 1e-12,
    transpose: bool = False,
) -> torch.Tensor:
    x = w.mT if transpose else w
    nn.init.normal_(x)
    x.div_(torch.linalg.vector_norm(x, dim=0, keepdim=True).clamp_min_(eps)).mul_(radius * math.sqrt(x.size(0)))
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
    x.div_(torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min_(eps)).mul_(radius / math.sqrt(x.size(1)))
    return w


@torch.no_grad()
def init_spectral_(w: torch.Tensor, radius: float = 1.0, input_like: bool = False) -> torch.Tensor:
    scale = math.sqrt(w.size(0) / w.size(1))
    if input_like:
        scale = max(1.0, scale)
    w_fp = w.data.double()
    nn.init.orthogonal_(w_fp)
    w_fp.mul_(radius * scale)
    w.data.copy_(w_fp.to(dtype=w.dtype))
    return w


@torch.no_grad()
def init_semiorthogonal_(w: torch.Tensor, radius: float = 1.0, input_like: bool = False) -> torch.Tensor:
    return init_spectral_(w, radius=radius, input_like=input_like)



def scion_transfer_lr(lr: float, mT: float = 1.0, mL: float = 1.0, alpha: float = 0.5):
    if lr <= 0.0:
        raise ValueError(f'invalid lr: {lr}')
    if mT <= 0.0:
        raise ValueError(f'invalid mT: {mT}')
    if mL <= 0.0:
        raise ValueError(f'invalid mL: {mL}')
    token_factor = mT ** -0.5
    depth_factor = mL ** (alpha - 1.0)
    return {
        'embed': lr * token_factor,
        'hidden': lr * token_factor * depth_factor,
        'out': lr * token_factor,
    }


class Scion(LionKCCWDPA):
    """Scion with optional fixed decoupled decay via `eta`."""

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
        nesterov: bool = True,
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


class ScionC(Scion):
    """Scion with corrected decay, typically driven by per-group `theta2`."""
