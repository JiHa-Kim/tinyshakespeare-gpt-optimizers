# ScionC Shrink/RMS Parametrization

This note documents the active optimizer coordinates used by
`scionc/train_shakespeare.py`:

- additive step-scale schedule `eta_t`,
- primal weight shrink schedule `sigma_t`,
- momentum-state retention schedule `beta_t`,
- target actual entrywise weight RMS `R_W`.

`eta_t` controls movement. `sigma_t` controls contraction of the previous primal
weight iterate. `R_W` is used directly for initialization, reporting, and the
capped one-step solve.

## Update

For one weight block, let `w` be the parameter, `g` the gradient, `m` the
momentum state, and `v` the ULMO atom. One optimizer step is:

```math
m'=\beta m+(1-\beta)g,
```

```math
v=\operatorname{ulmo}(m'),
```

```math
w'=\sigma_t w+\eta_t v.
```

The two clocks are specified as half-lives in processed-token units:

```math
\beta = 2^{-\Delta\tau/h_\beta},
\qquad
\sigma_{\mathrm{peak}} = 2^{-\Delta\tau/h_{\mathrm{shrink}}}.
```

The default simple schedule uses the WSD scalar `q_t=s_t/s_peak` for movement
and for the shrink halving exponent:

```math
\eta_t=q_t\eta_{\mathrm{peak}},
\qquad
\sigma_t=\sigma_{\mathrm{peak}}^{q_t}.
```

This keeps one visible schedule while preserving independent coordinates.

At the reference count increment `64 * 256`, the defaults are:

| group | target RMS `R_W` | `eta_peak` | `sigma_peak` |
|---|---:|---:|---:|
| embed | 0.700 | 0.035 | 0.965000 |
| hidden | 0.051 | 0.035 | 0.988333 |
| out | 0.022 | 0.035 | 0.996500 |

## One-Step RMS Solve

By default, the optimizer solves for the nonnegative `eta` that targets `R_W`
at the end of the current step:

```math
v_{\mathrm{sq}}\eta^2
+2\sigma s\eta
+\sigma^2\|w\|_{\mathrm{rms}}^2
-R_W^2
=0,
```

where:

```math
s=\langle w,v\rangle_{\mathrm{rms}},
\qquad
v_{\mathrm{sq}}=\|v\|_{\mathrm{rms}}^2.
```

The admissible root is capped by the baseline schedule:

```math
\eta_t=\min(\eta_t^{\mathrm{solve}},\eta_{\mathrm{lr},t}).
```

Eval lines print `weight_rms current/target`; this is the direct check that the
actual weights are following the scheduled RMS coordinate.

## Transfer

When batch size, block size, or gradient accumulation changes, keep the semantic
schedules fixed in processed-token units:

```math
H_\beta(\tau,\Delta\tau),
\qquad
H_{\mathrm{shrink}}(\tau,\Delta\tau),
\qquad
\eta(\tau),
\qquad
R_W(\tau).
```

Then recompute the per-update factors from the new count increment:

```math
\Delta\tau
=
\text{batch size}
\cdot
\text{block size}
\cdot
\text{gradient accumulation},
```

```math
\beta=2^{-H_\beta(\tau,\Delta\tau)},
\qquad
\sigma=2^{-H_{\mathrm{shrink}}(\tau,\Delta\tau)}.
```

## CLI

Primary coordinates:

- `--log2-step-scale*` / `--step-scale*`: peak eta multiplier.
- `--min-step-scale*`: eta schedule floor.
- `--shrink-half-life-*`: group shrink half-lives.
- `--beta-half-life`: momentum-state retention half-life.
- `--target-rms-*`: target actual entrywise weight RMS values. Defaults are
  `embed=0.70`, `hidden=0.051`, `out=0.022`.
- `--rms-solve` / `--no-rms-solve`: toggle capped one-step RMS targeting.
