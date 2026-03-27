# Vanilla Scion + branch-aware WSD LR tuning

This repo is organized around one goal: tune the peak learning rate for **vanilla Scion** under a **warmup-stable-decay / stable-decay** schedule without confusing the LR with the decay start.

Files:

- `lionk_ccwd.py`: Lion-K core with optional corrected decoupled decay
- `scion.py`: Scion LMOs, geometry-matched initialization helpers, LR transfer helper, and workspace reuse in the spectral LMO
- `gpt.py` + `train_shakespeare.py`: a small GPT training loop for tiny Shakespeare, with val-only evals during training and best-or-final checkpointing
- `tune_lr.py`: a branch-aware LR tuner for WSD / stable-decay, with cached batch sources and reusable tail runners

## Main policy

Default optimizer and tuning regime:

- optimizer: `scion`
- warmup: `0`
- min LR: `0`
- peak LR tuned on a proxy model first
- decay start tuned separately from the peak LR

Keep these fixed while tuning LR:

- `--rho-embed 1`
- `--rho-hidden 3`
- `--rho-out 10`
- `--phi 0.0`
- `--grad-clip 0.0`

Tune separately for:

- `--prenorm rmsnorm`
- `--prenorm rmsball`

## Why the tuner changed

A flat two-stage sweep with a single fixed WSD schedule is not enough for WSD.

The old logic effectively promoted the **largest stable LR** under one fixed decay fraction. That mixes together two different decisions:

1. peak LR
2. decay start / decay fraction

The new tuner treats WSD the way it is meant to be used:

1. **screen** peak LRs with a short constant-LR proxy run
2. **auto-expand** the search range if the best screen point sits on the search boundary
3. **shortlist** a few promising LRs
4. for each shortlisted LR, run **one constant-LR trunk**
5. branch from that trunk into several **decay tails** with different `decay_frac`
6. choose:
   - the best `(lr, decay_frac)` pair for a fixed budget
   - a promoted LR using an epsilon frontier over the best pair score per LR
   - a robust LR using an epsilon frontier over the median score across decay fractions

This gives you a much better quality/compute tradeoff than rerunning full WSD schedules for every LR.

## Geometry-matched Scion init

Initialization now matches the optimizer geometry instead of using generic unscaled init:

- token embedding: column-normalized init on the transposed embedding matrix, with `rho_embed`
- hidden matrices: spectral / semi-orthogonal init with the same dimension-aware scaling used by the spectral LMO, with `rho_hidden`
- output head: row-normalized init with `rho_out`

That keeps the model's initial parameter geometry aligned with the Scion update geometry.

## Exact single-run schedule

`train_shakespeare.py` now uses an exact schedule:

- warmup steps are explicit
- stable phase length is explicit
- if `decay_frac = 0`, there is no accidental one-step decay
- the last decay step reaches `min_lr` exactly

Default single-run decay fraction:

- `--decay-frac 0.285`

## Branch-aware tuning algorithm

### Stage A: screen

Use a short constant-LR run to cheaply locate the useful LR region.

Defaults:

- `screen_steps = 200`
- `screen_eval_interval = 50`
- `screen_eval_iters = 10`

The screen range auto-expands upward if the highest-tested LR is still stable and near the best screen score.

### Stage B: branch families

For each shortlisted LR, run one constant-LR trunk of length `branch_steps`, save checkpoints at the start of several decay tails, and then run those tails separately.

Default branch settings:

- `branch_steps = 600`
- `decay_fracs = 0.10,0.20,0.285,0.40`

Each decay fraction corresponds to a branch point:

- `branch_start = branch_steps - round(decay_frac * branch_steps)`

### Stage C: recommendations

The tuner produces three recommendations per prenorm:

1. `best_pair`
   - the best `(lr, decay_frac)` pair for the fixed proxy budget
2. `promoted_lr`
   - among LRs whose **best pair score** is within a small tolerance of the best, pick the **largest LR**
3. `robust_lr`
   - among LRs whose **median score across decay fractions** is near-best, pick the **largest LR**

This lets you separate:

- the best fixed-budget proxy schedule pair
- the LR you want to transfer upward and retune the decay start around

### Stage D: confirmation

The tuner reruns the most promising LRs across multiple seeds before printing final recommendations.

Default:

- `confirm_num_seeds = 3`

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
  --rho-embed 1 --rho-hidden 3 --rho-out 10 \
  --no-compile
```

### Branch-aware LR tuning

```bash
python tune_lr.py \
  --prenorm both \
  --exp2-min -14 --exp2-max -8 \
  --coarse-step 1.0 \
  --fine-radius 1.0 --fine-step 0.25 \
  --screen-steps 200 \
  --branch-steps 600 \
  --decay-fracs 0.10,0.20,0.285,0.40 \
  --confirm-num-seeds 3 \
  --batch-size 16 --grad-accum 4 --block-size 256 \
  --optimizer scion \
  --warmup-frac 0.0 --min-lr 0.0 \
  --rho-embed 1 --rho-hidden 3 --rho-out 10 \
  --no-compile
```

### Sweep the full model directly

```bash
python tune_lr.py \
  --no-proxy \
  --n-layer 6 --n-head 6 --d-model 384 \
  --max-iters 3000
```

## Output CSVs

`tune_lr.py` writes:

- `*_screen.csv`
- `*_branch_raw.csv`
- `*_pair_summary.csv`
- `*_lr_summary.csv`
- `*_confirm_raw.csv`
- `*_final_pair_summary.csv`
- `*_final_lr_summary.csv`
- `*_recommendations.csv`

## How to read the recommendations

- Use `best_pair` when you want the strongest fixed-budget proxy answer.
- Use `promoted_lr` when you want one LR to transfer to a larger model and then retune the decay start around it.
- Use `robust_lr` when you want an LR that is less sensitive to the exact decay fraction.

## Notes

- The tuner assumes warmup is zero during LR tuning unless you override it.
- If the top screen candidate stays stable and keeps improving, the screen range expands automatically.
- The trainer uses one global LR, so the practical transfer recommendation is still the **transferred hidden LR**.
