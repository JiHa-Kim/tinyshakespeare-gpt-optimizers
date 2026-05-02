# Minimal Lion-K / ScionC Repo

A tiny reference implementation organized by category:

- `scionc/optim/`: general Lion-K core with corrected decay, cautious masking, primal averaging, and direct shrinkage.
- `scionc/ulmos/`: basic ULMOs, Gram-NS, streaming SVD, and SVD-filter helpers.
- `scionc/models/`: compact GPT model and tiny Shakespeare data utilities.
- `scionc/probes/`: optional line, convergence, and optimizer-step stats probes.
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
with group-specific ULMOs, steady-state radii, shrink half-lives, and
dimensionless step-scale schedules. The raw additive step $\eta_t$ is derived by
the corrected RMS steady-state rule:

```math
\eta_t
=
s_t\rho
\sqrt{
\frac{1-2^{-2\Delta\tau/h_\zeta}}{S}
}.
```

`--readout-mu` controls the dimensionless Nesterov readout blend. Shrinkage is
controlled by `--shrink-half-life*`. Geometry-matched initialization uses the
same steady-state radius $\rho$. The momentum factor is:

```math
S
=
\frac{1+\beta}
{(1-\mu\beta)^2(1+\beta)+(\mu\beta)^2(1-\beta)}.
```

See [docs/scionc_steady_state_parametrization.md](docs/scionc_steady_state_parametrization.md)
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
- peak step scale: 1
- step-scale decay floor: 0
- derived additive step: group-specific, printed as `eta`
- steady-state radii: embedding 1, hidden 3, output 10
- readout mu: 1
- EMA half-life: about 1.56e5 processed tokens
- shrink half-lives:
  - embedding: about 3.19e5 processed tokens
  - hidden: about 9.68e5 processed tokens
  - output: about 3.24e6 processed tokens
- WSD schedule: 100 warmup steps, stable phase, 15% decay by default

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
  --prenorm rmsnorm \
  --batch-size 64 --grad-accum 1 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --step-scale 1 --min-step-scale 0 \
  --rho-embed 1 --rho-hidden 3 --rho-out 10 \
  --warmup-iters 100 --decay-frac 0.15 \
  --beta-half-life 1.565e5 --readout-mu 1 \
  --hidden-ulmo gram-ns \
  --embed-ulmo colnorm --out-ulmo sign \
  --shrink-half-life-embed 3.188e5 \
  --shrink-half-life-hidden 9.677e5 \
  --shrink-half-life-out 3.239e6
```

## Previous Reference Result

The previous WSD recipe reached validation loss `1.3912` over 200 eval batches
after 2k steps on tiny Shakespeare with batch size 64, gradient accumulation 1,
and block size 256. On a 4070 Ti, the compiled run reserved about 1.85 GB CUDA
memory and trained at about 450k tokens/s.

Example sample from `out/scionc_wsd_lr0p035_min0.pt`, using temperature `0.8`
and top-k `40`:

```text
To be, or not to be contented
You in conscience, your brother and your grace,
To make the people measure of these men.

NORTHUMBERLAND:
Your crown, sir, your body will confess
I should soldier till you weep the season,
And your tongue and yet I should be so;
Or in that bear the town that Clarence shall be loud.

BUCKINGHAM:
Why, bear the warriors with the world my life,
Who was an oath desires himself for stars.
```
