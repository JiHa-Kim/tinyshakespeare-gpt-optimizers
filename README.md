# Minimal Lion-K / ScionC repo

A tiny reference implementation with three pieces:

- `lionk_ccwd.py`: general Lion-K with corrected cautious weight decay and primal averaging
- `scion.py`: ScionC specialization, LMOs, init helpers, and simple transfer helper
- `gpt.py` + `train_shakespeare.py`: modern minimal GPT for tiny Shakespeare

Model choices:

- no bias
- gainless pre-norm blocks with an explicit ablation: `rmsnorm` or `rmsball`
- RoPE attention
- RMS-ball projection on `q` and `k`
- SwiGLU MLP
- untied embeddings / output head
- output head uses sign geometry; RowNorm is intentionally avoided
- ScionC defaults to primal averaging OFF (`phi = 0.0`)

Files:

- `lionk_ccwd.py`: Lion-K core with optional corrected decoupled decay
- `scion.py`: Scion LMOs, geometry-matched initialization helpers, LR transfer helper, Gram Newton-Schulz baseline LMO, and streaming SVD spectral LMO
- `gpt.py` + `train_shakespeare.py`: a small GPT training loop for tiny Shakespeare, with val-only evals during training and best-or-final checkpointing

## Main policy

Default optimizer settings:

- optimizer: `scionc`
- warmup: `0`
- min LR: `0`

Tune separately for:

- `--prenorm rmsnorm`
- `--prenorm rmsball`

## Geometry-matched Scion init

Initialization matches the Scion optimizer geometry instead of using generic unscaled init:

- token embedding: column-normalized init on the transposed embedding matrix, with `rho_embed`
- hidden matrices: spectral / semi-orthogonal init with the same dimension-aware scaling used by the spectral LMO, with `rho_hidden`
- output head: sign init with `rho_out`

## Hidden LMOs

The default hidden-matrix LMO is now `--hidden-lmo streaming-svd`, which keeps a per-parameter cached right-singular basis and applies one streaming power-iteration step per optimizer update.

- ColNorm before Cholesky
- direct Gram formation from `(M @ V).T @ (M @ V)` for numerical stability rather than `V.T @ (M.T @ M @ V)`
- one shifted CholeskyQR correction for the final QR
- RMS-offdiagonal gated QR refresh of the cached basis; default check interval is `100`
- no Householder QR fallback in the per-step hot path

There is also an experimental hidden-only closed-form filter:

- `--hidden-lmo svd-filter`
- defaults to `--filter-metric grad-sigma`, a zero-hook proxy using `sigma_i^2` as the denominator metric
- `--filter-metric full` restores the exact incoming activation covariance for hidden linear layers
- applies the closed-form diagonal filter under the same activation-perturbation budget as the matrix-sign direction
- uses `--filter-ridge` as the relative damping in `A = X^T X / b + lambda I`
- with `full`, `--filter-cov-interval` can reuse streamed activation covariances for speed
- with `full`, the MLP input covariance is shared between SwiGLU `gate` and `up`
- recommended faster-quality tradeoff: `--hidden-lmo svd-filter --filter-metric grad-sigma --spi-refresh-interval 100`

Use `--hidden-lmo gram-ns` for the baseline. This is the Gram Newton-Schulz form from Dao-AILab's implementation: five Polar Express coefficient steps with the 1.05 safety factor, a Gram-space update for rectangular matrices, a reset at iteration 2, and a cheap two-moment spectral upper-bound normalization from the already-formed Gram.

## Exact single-run schedule

`train_shakespeare.py` now uses an exact schedule:

- warmup steps are explicit
- stable phase length is explicit
- if `decay_frac = 0`, there is no accidental one-step decay
- the last decay step reaches `min_lr` exactly

Default single-run decay fraction:

- `--decay-frac 0.285`

## Recommended commands

### Train a single run

```bash
python train_shakespeare.py \
  --mode train \
  --optimizer scion \
  --prenorm rmsnorm \
  --batch-size 16 --grad-accum 4 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --lr 1e-3 \
  --warmup-frac 0.0 --decay-frac 0.285 --min-lr 0.0 \
  --beta2 0.95 --phi 0.0 \
  --hidden-lmo streaming-svd \
  --rho-embed 1 --rho-hidden 3 --rho-out 10 \
  --no-compile
```
