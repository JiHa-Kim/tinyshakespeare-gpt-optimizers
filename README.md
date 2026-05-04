# Minimal Lion-K / ScionC Repo

A compact ScionC sandbox organized by category:

- `scionc/optim/`: ScionC plus lower-level Lion-K experiments.
- `scionc/ulmos/`: basic ULMOs, Gram-NS, streaming SVD, and SVD-filter helpers.
- `scionc/models/`: compact GPT model and tiny Shakespeare data utilities.
- `scionc/probes/`: convergence, line, and optimizer-step stats probes.
- `scionc/train_shakespeare.py`: training entrypoint.

## Active ScionC Recipe

One optimizer update advances the count by
`batch_size * block_size * grad_accum` processed tokens. For one optimizer
group:

```math
m'=\beta m+(1-\beta)g,
\qquad
v=\operatorname{ulmo}(m'),
\qquad
w'=\zeta_t w+\eta_t v.
```

The tuned coordinates are the additive step-scale schedule `eta_t`, the
momentum-state retention half-life, the weight-retention half-life, and an
optional target RMS weight radius `R_W`:

```math
\beta=2^{-\Delta\tau/h_\beta},
\qquad
\zeta_{\mathrm{peak}}=2^{-\Delta\tau/h_\zeta}.
```

The default WSD scalar `q_t=s_t/s_peak` schedules the two independent
coordinates with one simple visible schedule:

```math
\eta_t=q_t\eta_{\mathrm{peak}},
\qquad
\zeta_t=\zeta_{\mathrm{peak}}^{q_t}.
```

The implied steady-state RMS radius is diagnostic and includes the actual
entrywise RMS scale of the ULMO atom:

```math
A_\zeta=\frac{1+\zeta_t\beta}{1-\zeta_t\beta},
\qquad
R_{\mathrm{ss},t}
=
\eta_t
\sqrt{
\frac{A_\zeta\,\|v_t\|_{\mathrm{rms}}^2}{1-\zeta_t^2}
}.
```

The script stores `eta_t` as the optimizer-group `lr` and stores `zeta_t` as
`weight_retention`. By default, the optimizer applies the one-step RMS-targeting
root capped by this baseline schedule; use `--no-rms-solve` to run the plain
step-scale schedule. Eval lines print `weight_rms current/target` for each
optimizer group.

See [docs/scionc_rms_radius_parametrization.md](docs/scionc_rms_radius_parametrization.md)
for the derivation.

The lower-level Lion-K/CCWD optimizer uses the same retention/radius language:
`lr` is the additive direction scale, `weight_retention` is the active
coordinate retention, and `rms_radius` derives the retention from the stationary
RMS balance. See
[docs/lionk_ccwd_rms_parametrization.md](docs/lionk_ccwd_rms_parametrization.md).

## Defaults

- optimizer: ScionC
- hidden ULMO: Gram Newton-Schulz
- input/output ULMOs: untied ColNorm + Sign; tied Sign + Sign
- batch size: 64
- gradient accumulation: 1
- block size: 256
- target actual RMS radii: embedding 0.70, hidden 0.051, output 0.022
- peak eta: 0.035 at step scale 1 (`log2=0`)
- legacy support diagnostics: embedding 1, hidden 3, output 10
- RMS solve: enabled, capped by the eta schedule
- schedule floor: 0
- readout mu: 1
- momentum-state retention half-life: about 2.21e5 processed tokens
- weight-retention half-lives:
  - embedding: about 3.19e5 processed tokens
  - hidden: about 9.68e5 processed tokens
  - output: about 3.24e6 processed tokens
- WSD schedule: 100 warmup steps, stable phase, 15% decay by default

## Hidden ULMOs

`--hidden-ulmo gram-ns` is the default. It uses the Gram Newton-Schulz form with
the four-moment spectral upper-bound normalization that reuses the first `G @ G`
product already needed by the polynomial iteration.

`--hidden-ulmo streaming-svd` keeps a per-parameter cached right-singular basis
and applies one or more streaming subspace steps per optimizer update.

`--hidden-ulmo svd-filter` adds an experimental diagonal filter using the exact
incoming activation covariance for hidden linear layers.

## Recommended Command

```bash
uv run python -m scionc.train_shakespeare \
  --mode train \
  --out-path out/scionc_rms_radius_2k.pt \
  --sample-out out/scionc_rms_radius_2k_samples.md \
  --prenorm rmsnorm \
  --batch-size 64 --grad-accum 1 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --log2-step-scale 0 --min-step-scale 0 \
  --warmup-iters 100 --decay-frac 0.15 \
  --beta-half-life 2.214e5 --readout-mu 1 \
  --hidden-ulmo gram-ns \
  --embed-ulmo colnorm --out-ulmo sign \
  --weight-retention-half-life-embed 3.188e5 \
  --weight-retention-half-life-hidden 9.677e5 \
  --weight-retention-half-life-out 3.239e6 \
  --temperature 0.8 --top-k 40 --sample-count 2
```

Evaluate the saved checkpoint with more batches:

```bash
uv run python -m scionc.train_shakespeare \
  --mode eval \
  --device cuda \
  --out-path out/scionc_rms_radius_2k.pt \
  --eval-iters 200
```
