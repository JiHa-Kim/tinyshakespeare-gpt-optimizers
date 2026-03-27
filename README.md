# Minimal Lion-K / ScionC repo

A tiny reference implementation with four pieces:

- `lionk_ccwd.py`: general Lion-K with corrected cautious weight decay and primal averaging
- `scion.py`: Scion specialization, LMOs, init helpers, and transfer helper
- `gpt.py` + `train_shakespeare.py`: modern minimal GPT for tiny Shakespeare
- `tune_lr.py`: minimal LR sweep script

Model choices:

- no bias
- gainless pre-norm blocks with an explicit ablation: `rmsnorm` or `rmsball`
- RoPE attention
- RMS-ball projection on `q` and `k`
- SwiGLU MLP
- untied embeddings / output head
- output head always uses row norm
- ScionC defaults to primal averaging OFF (`phi = 0.0`)
- ScionC defaults to no gradient clipping (`--grad-clip 0.0`)

## Tuning policy

Tune a single global learning rate first, but tune it **separately** for:

- `--prenorm rmsnorm`
- `--prenorm rmsball`

Keep the Scion radii fixed:

- `--rho-hidden 50`
- `--rho-out 3000`

Keep the rest fixed while tuning LR:

- `--phi 0.0`
- `--grad-clip 0.0`

## WSD schedule

The repo uses warmup-stable-decay learning rate schedule.

The LR at step `t` is:

- linear warmup for the first `warmup_steps`
- constant plateau at the peak LR
- linear decay over the last `decay_steps`

Main knobs:

- `--warmup-frac`
- `--warmup-iters`
- `--decay-frac`
- `--min-lr`

Recommended default for LR tuning:

- `--warmup-frac 0.0`
- `--decay-frac 0.2`
- `--min-lr 0.0`

That means:

- no warmup by default
- hold the peak LR for the first `80%` of training
- linearly decay to zero over the last `20%`

## Proxy-model LR tuning

By default, `tune_lr.py` tunes on a proxy model unless you explicitly override the model size or pass `--no-proxy`.

Default proxy settings:

- `--n-layer 4`
- `--n-head 4`
- `--d-model 256`
- `--max-iters 600`
- `--eval-interval 50`
- `--eval-iters 20`

## LR strategy with WSD

The sweep is two-stage in base-2 space:

1. coarse sweep over `log2(lr)`
2. fine sweep around the best coarse exponent

Selection rule:

- choose the **largest LR that stays stable**
- stability means no non-finite train/eval losses and no evaluation loss blow-up beyond `--diverge-mult x initial_val`
- only if no candidate is stable, fall back to best validation loss among finite runs

This is more aggressive than choosing pure best validation loss, which tends to bias conservative on short proxy WSD runs.

## Automatic proxy-to-target LR transfer

`tune_lr.py` automatically computes Scion transfer from the proxy run to the target run.

The transfer uses:

- token multiplier `mT`
- layer multiplier `mL`
- transfer exponent `alpha` (default `0.5`)

It prints:

- proxy LR selected by the sweep
- transferred target embed LR
- transferred target hidden LR
- transferred target output LR

Because the current train script uses one global LR, the practical recommendation is to start from the transferred hidden LR.

## Install

```bash
pip install torch
```

## Train

```bash
python train_shakespeare.py --mode train --compile
```

Useful flags:

```bash
python train_shakespeare.py \
  --mode train \
  --n-layer 6 --n-head 6 --d-model 384 \
  --batch-size 16 --grad-accum 4 --block-size 256 \
  --lr 1e-3 \
  --warmup-frac 0.0 --decay-frac 0.2 --min-lr 0.0 \
  --beta2 0.95 --phi 0.0 \
  --rho-hidden 50 --rho-out 3000 \
  --prenorm rmsnorm \
  --compile
```

RMS-ball pre-norm ablation:

```bash
python train_shakespeare.py --mode train --prenorm rmsball
```

To turn primal averaging on explicitly:

```bash
python train_shakespeare.py --mode train --phi 1.0
```

## LR sweep

Proxy sweep with default proxy model and WSD:

```bash
python tune_lr.py \
  --exp2-min -14 --exp2-max -8 \
  --coarse-step 1.0 \
  --fine-radius 1.0 --fine-step 0.25 \
  --csv-path out/lr_sweep.csv \
  --batch-size 16 --grad-accum 4 --block-size 256 \
  --warmup-frac 0.0 --decay-frac 0.2 --min-lr 0.0 \
  --compile
```

This runs both:

- `prenorm = rmsnorm`
- `prenorm = rmsball`

and prints separate recommendations for each.

To disable the proxy and sweep the full model directly:

```bash
python tune_lr.py --no-proxy --n-layer 6 --n-head 6 --d-model 384
```

## Logging

When `--compile` is enabled, the script logs:

- `compile_seconds`: one-time compile and first forward/backward warmup cost
- `train_seconds`: elapsed training time excluding compile
- `tok/s`: throughput excluding compile

The training loop also returns divergence information used by `tune_lr.py`:

- `diverged`
- `diverge_reason`
- `initial_val`
- `max_val`

## Sample

```bash
python train_shakespeare.py --mode sample --out-path out/scionc_shakespeare.pt --prompt "To be, or not to be"
```

## File guide

### `lionk_ccwd.py`

- `LionKCCWDPA`: general optimizer core
- `corrected_eta(...)`: corrected multiplicative decay helper

### `scion.py`

- LMOs with explicit radii: `ColNormLMO`, `SpectralLMO`, `RowNormLMO`, `RMSLMO`
- init helpers: `init_colnorm_`, `init_rownorm_`, `init_semiorthogonal_`
- transfer helper: `scion_transfer_lr(...)`
- optimizer specialization: `Scion`

### `gpt.py`

- GPT model
- gainless `Norm(kind='rmsnorm'|'rmsball')`
- tiny Shakespeare downloader
- on-device vectorized dataset batching

### `train_shakespeare.py`

- Scion init and parameter grouping
- one-global-lr tuning interface
- WSD-only LR schedule
- explicit radii flags for hidden/output LMOs
- gradient accumulation
- compile-time warmup and separate logging
- train loop
- checkpoint save/load
- sampling

### `tune_lr.py`

- two-stage base-2 LR sweep
- proxy-model defaults
- CSV export
- ranking by best validation loss
