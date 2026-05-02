# Minimal Lion-K / ScionC Repo

A tiny reference implementation organized by category:

- `scionc/optim/`: general Lion-K core with corrected decay, cautious masking, primal averaging, and direct shrinkage.
- `scionc/lmos/`: ScionC wrapper, basic LMOs, Gram-NS, streaming SVD, SVD-filter, and geometry-matched init helpers.
- `scionc/models/`: compact GPT model and tiny Shakespeare data utilities.
- `scionc/probes/`: optional line, convergence, and optimizer-step stats probes.
- `scionc/train_shakespeare.py`: training entrypoint.

## Active Recipe

The Shakespeare training script uses ScionC in token half-life coordinates.
One optimizer update advances the count by
`batch_size * block_size * grad_accum` processed tokens.

For optimizer group `i`, the LMO returns a unit atom:

$$
V_{i,t}=\operatorname{LMO}_{g_i}(M_{i,t}),
\qquad
\|V_{i,t}\|_{g_i}=1.
$$

EMA memory and direct shrinkage are multiplicative actions:

$$
\beta_t=2^{-\Delta\tau/h_\beta},
\qquad
a_{i,t}=2^{-\Delta\tau/h_{a,i}}.
$$

The step applies direct shrinkage and the additive LR separately:

$$
M_{i,t+1}=\beta_t M_{i,t}+(1-\beta_t)G_{i,t},
\qquad
X_{i,t+1}=a_{i,t}X_{i,t}+\alpha_{i,t}V_{i,t}.
$$

The scheduled learning rate is the additive scale $\alpha_t$. `--readout-mu`
controls the dimensionless Nesterov readout blend. The `--rho-*` values are
geometry-matched initialization radii only; shrinkage is controlled by
`--shrink-half-life*`.

The lower-level optimizer still exposes corrected decay, cautious masking, and
primal averaging for experiments. The Shakespeare recipe always supplies direct
`shrink` factors, so those decay paths are not part of this training entrypoint.

Current defaults:

- optimizer: ScionC
- hidden LMO: Gram Newton-Schulz
- input embedding LMO: ColNorm
- output head LMO: Sign
- batch size: 64
- gradient accumulation: 1
- block size: 256
- peak LR: 0.035
- decay floor: 0
- readout mu: 1
- EMA half-life: about 1.56e5 processed tokens
- shrink half-lives:
  - embedding: about 3.19e5 processed tokens
  - hidden: about 9.68e5 processed tokens
  - output: about 3.24e6 processed tokens
- WSD schedule: 100 warmup steps, stable phase, 15% decay by default

Default radii:

| Group | Geometry | Radius |
|---|---|---:|
| input embedding | ColNorm | 1 |
| hidden matrices | spectral | 3 |
| output head | Sign | 10 |

## Hidden LMOs

`--hidden-lmo gram-ns` is the default. It uses the Gram Newton-Schulz form:
five minimax Polar Express coefficient steps with an fp64-derived final
normalization so the composed scalar map satisfies `p(1)=1`, the 1.05 safety
factor, a Gram-space rectangular update, a reset at iteration 2, and a cheap
two-moment spectral upper-bound normalization from the already-formed Gram.

`--hidden-lmo streaming-svd` keeps a per-parameter cached right-singular basis
and applies one or more streaming subspace steps per optimizer update.

`--hidden-lmo svd-filter` adds an experimental diagonal filter using the exact
incoming activation covariance for hidden linear layers.

## Recommended Command

```bash
uv run python -m scionc.train_shakespeare \
  --mode train \
  --prenorm rmsnorm \
  --batch-size 64 --grad-accum 1 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --lr 3.5e-2 --min-lr 0 \
  --warmup-iters 100 --decay-frac 0.15 \
  --beta-half-life 1.565e5 --readout-mu 1 \
  --hidden-lmo gram-ns \
  --embed-lmo colnorm --out-lmo sign \
  --rho-embed 1 --rho-hidden 3 --rho-out 10 \
  --shrink-half-life-embed 3.188e5 \
  --shrink-half-life-hidden 9.677e5 \
  --shrink-half-life-out 3.239e6
```

## Current Result

The current working recipe reached validation loss `1.3912` over 200 eval
batches after 2k steps on tiny Shakespeare with batch size 64, gradient
accumulation 1, and block size 256. On a 4070 Ti, the compiled run reserved
about 1.85 GB CUDA memory and trained at about 450k tokens/s.

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
