# Minimal Lion-K / ScionC Repo

A tiny reference implementation organized by category:

- `scionc/optim/`: general Lion-K core with corrected decay, cautious masking, primal averaging, and direct shrinkage.
- `scionc/ulmos/`: basic ULMOs, Gram-NS, streaming SVD, and SVD-filter helpers.
- `scionc/models/`: compact GPT model and tiny Shakespeare data utilities.
- `scionc/probes/`: optional line and optimizer-step stats probes.
- `scionc/train_shakespeare.py`: training entrypoint.

## Active Recipe

The Shakespeare training script uses ScionC in token half-life coordinates.
One optimizer update advances the count by
`batch_size * block_size * grad_accum` processed tokens.

For one optimizer group, write the parameters, gradients, and memory as $X$,
$G$, and $M$. The memory and Nesterov-style readout are:

```math
M \leftarrow \beta M+(1-\beta)G,
\qquad
R \leftarrow (1-\mu)G+\mu M.
```

EMA memory and direct shrinkage are multiplicative actions:

```math
\beta=2^{-\Delta\tau/h_\beta},
\qquad
\zeta=2^{-\Delta\tau/h_\zeta}.
```

The group ULMO returns a unit atom, and the step applies direct shrinkage and
the additive step separately:

```math
V \leftarrow \operatorname{ULMO}(R),
\qquad
X \leftarrow \zeta X+\eta V.
```

The same equations apply independently to embedding, hidden, and output groups,
with group-specific ULMOs, reference radii, shrink half-lives, and
dimensionless step-scale schedules. The raw additive step $\eta_t$ is derived by
the geometry invariant rule:

```math
\zeta_t=\zeta_0^{r_t},
\qquad
\eta_t
=
s_{\mathrm{peak}}(1-\zeta_t)\rho.
```

Here $\zeta_0$ is the peak shrink factor from the shrink half-life, and $r_t$
is the per-update warmup/stable/decay schedule ratio. Equivalently, $r_t$ is
the interval-average shrink-rate multiplier for that optimizer update. With
this convention, $\zeta_t=\zeta_0^{r_t}$ and $\zeta_t\to1$ as the schedule
decays to zero.
With $s_{\mathrm{peak}}=1$, the radius ball is invariant under a persistent unit
ULMO atom: $\|X_{\mathrm{new}}\|\le\rho$ whenever $\|X\|\le\rho$. Peak tuning
uses the dimensionless coordinate $\ell=\log_2 s_{\mathrm{peak}}$, so changing
$\ell$ by 1 doubles or halves the derived additive step. The script accepts
`--log2-step-scale*` for that coordinate and keeps `--step-scale*` as a linear
compatibility override.

`--readout-mu` controls the dimensionless Nesterov readout blend. Shrinkage is
controlled by `--shrink-half-life*`. Geometry-matched initialization uses the
same reference radius $\rho$.

See [docs/scionc_reference_radius_parametrization.md](docs/scionc_reference_radius_parametrization.md)
for the derivation.

The lower-level optimizer still exposes corrected decay, cautious masking, and
primal averaging for experiments. The Shakespeare recipe always supplies direct
`shrink` factors, so those decay paths are not part of this training entrypoint.

Current defaults:

- optimizer: ScionC
- hidden ULMO: Gram Newton-Schulz
- input embedding ULMO: ColNorm
- output head ULMO: Sign
- batch size: 64
- gradient accumulation: 1
- block size: 256
- peak step scale: 1 (`log2=0`)
- schedule floor: 0
- peak derived additive steps at the default count increment:
  - embedding: 0.035
  - hidden: 0.035
  - output: 0.035
- reference radii: embedding 1, hidden 3, output 10
- readout mu: 1
- EMA half-life: about 2.21e5 processed tokens
- shrink half-lives:
  - embedding: about 3.19e5 processed tokens
  - hidden: about 9.68e5 processed tokens
  - output: about 3.24e6 processed tokens
- WSD schedule: 100 warmup steps, stable phase, 15% decay by default
- train-state checkpoint: saved at the decay start by default so tail schedules
  can be resumed with `--resume-state`; use `--stop-after-state-save` to build
  only the shared stable prefix

## Hidden ULMOs

`--hidden-ulmo gram-ns` is the default. It uses the Gram Newton-Schulz form:
five minimax Polar Express coefficient steps with an fp64-derived final
normalization so the composed scalar map satisfies `p(1)=1`, the 1.05 safety
factor, a Gram-space rectangular update, a reset at iteration 2, and a cheap
two-moment spectral upper-bound normalization from the already-formed Gram.

`--hidden-ulmo streaming-svd` keeps a per-parameter cached right-singular basis
and applies one or more streaming subspace steps per optimizer update.

`--hidden-ulmo svd-filter` adds an experimental diagonal filter using the exact
incoming activation covariance for hidden linear layers.

## Recommended Command

```bash
uv run python -m scionc.train_shakespeare \
  --mode train \
  --out-path out/scionc_rate_scheduled_2k.pt \
  --sample-out out/scionc_rate_scheduled_2k_t08_k40_samples.md \
  --prenorm rmsnorm \
  --batch-size 64 --grad-accum 1 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --log2-step-scale 0 --min-step-scale 0 \
  --rho-embed 1 --rho-hidden 3 --rho-out 10 \
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
  --out-path out/scionc_rate_scheduled_2k.pt \
  --eval-iters 200
```

## Current Reference Result

The current shrink-rate parametrization completed a 2k-step tiny Shakespeare
run with batch size 64, gradient accumulation 1, and block size 256. The final
training eval at step 1999 reported best validation loss `1.3987`. A separate
200-batch eval of `out/scionc_rate_scheduled_2k.pt` reported:

```text
train 1.0763 | val 1.4068
```

The run also wrote a branchable train-state checkpoint at
`out/scionc_rate_scheduled_2k_state_step01700.pt`, which is the start of the
decay phase.

Example sample from `out/scionc_rate_scheduled_2k_t08_k40_samples.md`, using
temperature `0.8` and top-k `40`:

```text
To be, or not to be excellent.

ROMEO:
So do you see! stir, if you say the Earl of Wiltshire,
As now to take an after young Peter's brother.

ROMEO:
Impossible honourable between the first,
Of blood will sue by England's friendship that love
In all the people.
```
