# Minimal Lion-K / ScionC Repo

A tiny reference implementation organized by category:

- `scionc/optim/`: general Lion-K core with optional cautious weight decay and primal averaging.
- `scionc/lmos/`: ScionC wrapper, basic LMOs, Gram-NS, streaming SVD, SVD-filter, and geometry-matched init helpers.
- `scionc/models/`: compact GPT model and tiny Shakespeare data utilities.
- `scionc/probes/`: optional line, convergence, and optimizer-step stats probes.
- `scionc/train_shakespeare.py`: training entrypoint.

## Active Recipe

The Shakespeare training script uses ScionC in standard Lion-K coordinates.

For optimizer group `i`, the LMO returns a unit atom:

$$
V_{i,t}=\operatorname{LMO}_{g_i}(M_{i,t}),
\qquad
\|V_{i,t}\|_{g_i}=1.
$$

The step applies the effective LR and decay:

$$
X_{i,t+1}=X_{i,t}+\alpha_{i,t}V_{i,t}-\alpha_{i,t}\lambda_i X_{i,t}.
$$

Constrained Scion radius $\rho_i$ is represented as $\lambda_i=1/\rho_i$.
The scheduled learning rate is the Lion-K effective LR $\alpha_t$.
`--beta2` controls the current EMA gradient proxy used by the LMO

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
- beta2: 0.93
- WSD schedule: 100 warmup steps, stable phase, 15% decay by default

The general Lion-K core still supports cautious weight decay. It is out of scope
for the ScionC Shakespeare recipe.

The general Lion-K core also supports primal averaging. It is out of scope for
the ScionC Shakespeare recipe.

Default radii:

| Group | Geometry | Radius | Decay |
|---|---|---:|---:|
| input embedding | ColNorm | 1 | 1 |
| hidden matrices | spectral | 3 | 0.333333 |
| output head | Sign | 10 | 0.1 |

## Hidden LMOs

`--hidden-lmo gram-ns` is the default. It uses the Gram Newton-Schulz form:
five minimax Polar Express coefficient steps with an fp64-derived final
normalization so the composed scalar map satisfies `p(1)=1`, the 1.05 safety
factor, a Gram-space rectangular update, a reset at iteration 2, and a cheap
two-moment spectral upper-bound normalization from the already-formed Gram.

`--hidden-lmo streaming-svd` keeps a per-parameter cached right-singular basis
and applies one or more streaming subspace steps per optimizer update.

`--hidden-lmo svd-filter` adds an experimental diagonal filter:

- `--filter-metric grad-sigma`: zero-hook proxy using singular values of the gradient proxy.
- `--filter-metric full`: exact incoming activation covariance for hidden linear layers.

## Recommended Command

```bash
uv run python -m scionc.train_shakespeare \
  --mode train \
  --prenorm rmsnorm \
  --batch-size 64 --grad-accum 1 --block-size 256 \
  --n-layer 6 --n-head 6 --d-model 384 \
  --lr 3.5e-2 --min-lr 0 \
  --warmup-iters 100 --decay-frac 0.15 \
  --beta2 0.93 \
  --hidden-lmo gram-ns \
  --embed-lmo colnorm --out-lmo sign \
  --rho-embed 1 --rho-hidden 3 --rho-out 10
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
