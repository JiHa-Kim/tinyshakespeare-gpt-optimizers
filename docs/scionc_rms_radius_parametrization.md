# ScionC Step-Scale Parametrization

This note documents the optimizer coordinates used by
`scionc/train_shakespeare.py`. The active coordinate system is:

- additive step-scale schedule `eta_t`,
- weight-retention half-life schedule,
- momentum-state retention half-life schedule,
- target actual entrywise RMS radius `R_W` for initialization, reporting, and
  the capped one-step solve.

The important change from the older radius form is that `eta_t` and `zeta_t`
are independent coordinates. The implied radius
`rho_t = eta_t / (1 - zeta_t)` is reported, but it is not the tuned coordinate.

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
w'=\zeta_t w+\eta_t v.
```

The retentions come from half-lives in processed-token units:

```math
\beta = 2^{-\Delta\tau/h_\beta},
\qquad
\zeta_{\mathrm{peak}} = 2^{-\Delta\tau/h_\zeta}.
```

The default simple schedule uses the WSD scalar `q_t=s_t/s_peak` for both
movement and the retention halving exponent:

```math
\eta_t=q_t\eta_{\mathrm{peak}},
\qquad
\zeta_t=\zeta_{\mathrm{peak}}^{q_t}.
```

This keeps one visible schedule while preserving independent coordinates. The
old direct-radius recipe is approximately recovered by choosing
`eta_peak ~= (1 - zeta_peak) rho_ref`; the defaults set
`eta_peak = 0.035` at `--log2-step-scale 0`.

At the reference count increment `64 * 256`, this gives the old support
diagnostics exactly. The default `R_W` values are actual entrywise RMS targets,
not support radii:

| group | `R_W` actual RMS | `eta_peak` | `zeta_peak` | support diagnostic `rho` | prior `R_ss` |
|---|---:|---:|---:|---:|---:|
| embed | 0.700 | 0.035 | 0.965000 | 1 | 0.640 |
| hidden | 0.051 | 0.035 | 0.988333 | 3 | 0.059 |
| out | 0.022 | 0.035 | 0.996500 | 10 | 0.0066 |

For ablations, `--weight-retention-schedule constant` keeps `zeta_t` at the
peak half-life value and schedules only `eta_t`.

## Steady-State RMS

Under locally constant retentions and step scales, using the EMA
atom-correlation prior and the actual atom RMS scale:

```math
A_\zeta = \frac{1+\zeta\beta}{1-\zeta\beta}.
```

The implied stationary RMS radius is:

```math
R_{\mathrm{ss}}
=
\eta
\sqrt{
\frac{A_\zeta\,\|v\|_{\mathrm{rms}}^2}{1-\zeta^2}
}.
```

The output schedule prints both the configured target `rw` and the implied
`rss`, so old step-scale quality can be compared against RMS-targeted
experiments without changing the primary learning-rate coordinate. Eval lines
also print `weight_rms current/target`, which is the direct check that `R_W`
is in the actual weight RMS coordinate.

## One-Step RMS Solve

By default, the optimizer solves for the nonnegative `eta` that targets `R_W`
at the end of the current step:

```math
v_{\mathrm{sq}}\eta^2
+2\zeta s\eta
+\zeta^2\|w\|_{\mathrm{rms}}^2
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

With `--no-rms-solve`, default initialization uses the legacy support radii and
`R_W` is used for reporting only, unless an explicit `--rms-radius-*` override
is supplied. This keeps the plain old-quality recovery path available: tune
`eta_t` directly, keep the retention clock in half-life units, and treat
`R_ss` as a diagnostic.

## Transfer

When batch size, block size, or gradient accumulation changes, keep the
semantic schedules fixed in processed-token units:

```math
H_\beta(\tau,\Delta\tau),
\qquad
H_\zeta(\tau,\Delta\tau),
\qquad
\eta(\tau).
```

Then recompute the per-update retentions from the new count increment:

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
\zeta=2^{-H_\zeta(\tau,\Delta\tau)}.
```

## CLI

Primary coordinates:

- `--log2-step-scale*` / `--step-scale*`: peak eta multiplier.
- `--min-step-scale*`: eta schedule floor.
- `--weight-retention-half-life-*`: group weight-retention half-lives.
- `--beta-half-life`: momentum-state retention half-life.
- `--rms-radius-*`: target actual entrywise RMS radii. Defaults are
  `embed=0.70`, `hidden=0.051`, `out=0.022`.
- `--rms-solve` / `--no-rms-solve`: toggle capped one-step RMS targeting.
- `--weight-retention-schedule`: `scheduled`, `constant`, or compatibility
  aliases.
