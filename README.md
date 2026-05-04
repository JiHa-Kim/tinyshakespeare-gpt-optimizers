# Minimal ScionC Repo

A compact ScionC sandbox organized by category:

- `scionc/optim/`: ScionC optimizer and schedule parametrization helpers.
- `scionc/ulmos/`: ULMOs, Gram-NS, and streaming SVD helpers.
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
w'=\sigma_t w+\eta_t v.
```

The active coordinates are:

- additive step scale `eta_t`,
- primal weight shrink `sigma_t`,
- momentum-state retention `beta`,
- target actual entrywise weight RMS `R_W`.

Half-lives are specified in processed-token units:

```math
\beta=2^{-\Delta\tau/h_\beta},
\qquad
\sigma_{\mathrm{peak}}=2^{-\Delta\tau/h_{\mathrm{shrink}}}.
```

The default WSD scalar `q_t=s_t/s_peak` keeps one simple visible schedule while
leaving movement and shrink independent:

```math
\eta_t=q_t\eta_{\mathrm{peak}},
\qquad
\sigma_t=\sigma_{\mathrm{peak}}^{q_t}.
```

By default, ScionC solves the nonnegative one-step `eta` that targets `R_W` at
the end of the update, capped by the baseline `eta_t` schedule. Eval lines print
`weight_rms current/target` for each optimizer group.

See [docs/scionc_shrink_rms_parametrization.md](docs/scionc_shrink_rms_parametrization.md)
for the derivation.

## Defaults

- optimizer: ScionC
- hidden ULMO: Gram Newton-Schulz
- input/output ULMOs: untied ColNorm + Sign; tied Sign + Sign
- batch size: 64
- gradient accumulation: 1
- block size: 256
- target actual weight RMS: embedding 0.70, hidden 0.051, output 0.022
- peak eta: 0.035 at step scale 1 (`log2=0`)
- RMS solve: enabled, capped by the eta schedule
- schedule floor: 0
- readout mu: 1
- momentum-state half-life: about 2.21e5 processed tokens
- shrink half-lives:
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

## Recommended Command

```bash
uv run python -m scionc.train_shakespeare \
  --mode train \
  --out-path out/scionc_target_rms_2k.pt \
  --sample-out out/scionc_target_rms_2k_samples.md \
  --prenorm rmsnorm \
  --batch-size 64 --grad-accum 1 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --log2-step-scale 0 --min-step-scale 0 \
  --warmup-iters 100 --decay-frac 0.15 \
  --beta-half-life 2.214e5 --readout-mu 1 \
  --hidden-ulmo gram-ns \
  --embed-ulmo colnorm --out-ulmo sign \
  --shrink-half-life-embed 3.188e5 \
  --shrink-half-life-hidden 9.677e5 \
  --shrink-half-life-out 3.239e6 \
  --temperature 0.8 --top-k 40 --sample-count 2
```

Evaluate the saved checkpoint with more batches:

```bash
uv run python -m scionc.train_shakespeare \
  --mode eval \
  --device cuda \
  --out-path out/scionc_target_rms_2k.pt \
  --eval-iters 200
```
