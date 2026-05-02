# ScionC Reference-Radius Parametrization

This note documents the active ScionC coordinates used by
`scionc/train_shakespeare.py`. The goal is to make the optimizer settings
transfer cleanly when batch size, block size, or gradient accumulation change.

The main design choice is to expose semantic coordinates:

- a reference radius for each ULMO geometry,
- token half-lives for EMA memory and direct shrinkage,
- a dimensionless peak step scale,
- and a schedule ratio for the shrink rate.

The PyTorch optimizer group still stores the derived additive step in `lr`, but
conceptually that value is $\eta_t$ in the equations below.

## One Group

For one optimizer group, let $X$ be the parameter tensor, $G$ the current
gradient, and $M$ the EMA memory. One optimizer update advances the count by
$\Delta\tau$ processed tokens:

```math
\Delta\tau
=
\text{batch size}
\cdot
\text{block size}
\cdot
\text{gradient accumulation}.
```

The half-life coordinates are converted to per-update factors:

```math
\beta
=
2^{-\Delta\tau/h_\beta},
\qquad
\zeta_0
=
2^{-\Delta\tau/h_\zeta}.
```

The memory update and readout are:

```math
M_{\mathrm{new}}
=
\beta M+(1-\beta)G,
\qquad
R
=
(1-\mu)G+\mu M_{\mathrm{new}}.
```

The ULMO returns a unit atom in the group geometry:

```math
V
=
\operatorname{ULMO}(R).
```

The optimizer step is:

```math
X_{\mathrm{new}}
=
\zeta_t X+\eta_t V.
```

`ScionC` initializes the memory buffer to zero on first use, updates it with the
current gradient, and then forms the readout above.

## Scheduled Shrink Rate

The training schedule produces a nonnegative per-update ratio $r_t$. In the WSD
setup, $r_t=1$ during the stable peak and decays to the configured floor during
the tail. The code obtains $r_t$ by scheduling the dimensionless step-scale
value and dividing by the peak step scale.

The active recipe schedules the continuous shrink rate:

```math
\zeta_t
=
\zeta_0^{r_t}
=
2^{-r_t\Delta\tau/h_\zeta}.
```

Equivalently, the exact continuous-rate expression is:

```math
\zeta_t
=
2^{-\frac{1}{h_\zeta}
\int_{\tau_t}^{\tau_t+\Delta\tau} r(s)\,ds}.
```

The discrete formula above is exact when $r_t$ is the interval average over the
update, or when the schedule is treated as piecewise constant during each
update. When $r_t\to0$, the per-step shrink factor tends to one and the shrink
action turns off.

## Geometry-Invariant Step

Let $\|\cdot\|$ be the primal norm declared by the group ULMO and let $\rho$ be
the reference radius. Since the ULMO atom satisfies $\|V\|\le1$:

```math
\|X_{\mathrm{new}}\|
\le
\zeta_t\|X\|+\eta_t.
```

The ball $\|X\|\le\rho$ is invariant under a persistent unit atom if:

```math
\eta_t
\le
(1-\zeta_t)\rho.
```

The active additive step is therefore:

```math
\eta_t
=
s_{\mathrm{peak}}(1-\zeta_t)\rho.
```

At $s_{\mathrm{peak}}=1$, this is the largest persistent-direction step that
preserves the radius ball at every schedule ratio. Values above one are allowed
as explicit over-relaxation; then $\rho$ remains the reference geometry scale,
but not a worst-case invariant bound.

The preferred tuning coordinate is additive in log space:

```math
s_{\mathrm{peak}}
=
2^{\ell_{\mathrm{peak}}}.
```

Changing $\ell_{\mathrm{peak}}$ by one doubles or halves the derived additive
step without changing the shrink half-life or radius.

## Default Calibration

The default count increment is:

```math
\Delta\tau
=
64\cdot256
=
16384.
```

The default reference radii are:

```math
\rho_{\mathrm{embed}}=1,
\qquad
\rho_{\mathrm{hidden}}=3,
\qquad
\rho_{\mathrm{out}}=10.
```

The shrink half-lives are chosen so that with $s_{\mathrm{peak}}=1$ and
$r_t=1$, every group has the same peak additive step $\eta_0=0.035$:

```math
\zeta_0
=
1-\frac{0.035}{\rho},
\qquad
h_\zeta
=
-\frac{\Delta\tau}{\log_2 \zeta_0}.
```

This gives the peak factors:

```math
\zeta_{\mathrm{embed}}=0.965,
\qquad
\zeta_{\mathrm{hidden}}\approx0.988333,
\qquad
\zeta_{\mathrm{out}}=0.9965.
```

The default EMA half-life is similarly calibrated so that the default
per-update memory factor is $\beta=0.95$ at $\Delta\tau=16384$:

```math
h_\beta
=
-\frac{16384}{\log_2 0.95}.
```

## Transfer Rule

When the count increment changes from $\Delta\tau$ to $\Delta\tau_\star$, keep
the semantic hyperparameters fixed:

```math
\rho_\star = \rho,
\qquad
h_{\zeta,\star} = h_\zeta,
\qquad
s_{\mathrm{peak},\star}=s_{\mathrm{peak}},
\qquad
r_{t,\star}=r_t,
\qquad
h_{\beta,\star}=h_\beta,
\qquad
\mu_\star=\mu.
```

Then recompute the raw factors from the new count increment:

```math
\beta_\star
=
2^{-\Delta\tau_\star/h_\beta},
\qquad
\zeta_{0,\star}
=
2^{-\Delta\tau_\star/h_\zeta},
\qquad
\zeta_{t,\star}
=
\zeta_{0,\star}^{r_t}.
```

The additive step follows from the same radius rule:

```math
\eta_{t,\star}
=
s_{\mathrm{peak}}(1-\zeta_{t,\star})\rho.
```

This preserves memory time, shrink-rate time, the reference radius, the
over-relaxation coordinate, and the schedule shape in processed-token units.

## Relation To Old LR-Coupled Numerics

The old raw-learning-rate recipe with weight decay $1/\rho$ can be written:

```math
X_{\mathrm{new}}
=
\left(1-\frac{\eta_t}{\rho}\right)X+\eta_t V.
```

If $\eta_t=r_t\eta_0$ and $\zeta_0=1-\eta_0/\rho$, the exact one-to-one
translation is:

```math
\eta_t
=
r_t\eta_0,
\qquad
\zeta_t
=
1-r_t(1-\zeta_0).
```

So the old WSD tail scheduled both the additive action and the shrink action.
Keeping $\zeta$ fixed while decaying only $\eta$ is a different, independent
shrink recipe.

The active ScionC recipe keeps the decoupled half-life coordinate and schedules
the rate instead:

```math
\zeta_t
=
\zeta_0^{r_t},
\qquad
\eta_t
=
s_{\mathrm{peak}}(1-\zeta_t)\rho.
```

This is not exactly the old discrete LR-coupled schedule, but it agrees to first
order in the small-shrink regime:

```math
1-\zeta_0^{r_t}
\approx
r_t(1-\zeta_0).
```

That makes it a principled rate-coordinate analogue rather than a hidden return
to raw learning-rate semantics.

## Corrected RMS Step

The lower-level Lion-K utilities still contain the corrected RMS balance used
by experiments that want a random-direction radius estimate. Assume the update
atoms have effective squared size:

```math
\mathbb{E}\|V\|^2
=
C.
```

For Lion-K style normalized updates, write:

```math
C
=
\frac{c_u^2 S}{q},
```

where $c_u^2$ is the atom squared-norm scale, $S$ is the momentum amplification
factor, and $q$ is the cautious-mask keep fraction. The second-moment steady
state at radius $\rho$ satisfies:

```math
\rho^2
=
\zeta^2\rho^2+\eta^2 C.
```

Solving for the additive step gives:

```math
\eta
=
s\rho
\sqrt{
\frac{q(1-\zeta^2)}
{c_u^2S}
}.
```

This RMS rule targets a second-moment steady-state radius. The active
reference-radius rule instead targets the persistent-direction invariant bound.
Under the RMS model, the active rule gives:

```math
\frac{R_{\mathrm{rms}}}{\rho}
=
s_{\mathrm{peak}}\sqrt{C}
\sqrt{\frac{1-\zeta}{1+\zeta}}.
```

Thus $\rho$ is not the RMS steady-state radius in the active parametrization;
it is the geometry reference radius for the invariant action.

In the small-shrink regime with $d=1-\zeta$:

```math
d
\approx
\frac{\eta^2 c_u^2S}{2q\rho^2}.
```

That is the existing corrected decoupled-decay approximation used by the
lower-level optimizer paths.

## User-Facing Coordinates

The primary CLI coordinates are:

- `--rho-*`: group reference radius $\rho$.
- `--shrink-half-life-*`: shrink half-life $h_\zeta$.
- `--log2-step-scale*`: log coordinate $\ell_{\mathrm{peak}}$.
- `--step-scale*`: linear compatibility coordinate $s_{\mathrm{peak}}$.
- `--min-step-scale*`: linear schedule floor; the rate floor is this value
  divided by the peak step scale.
- `--beta-half-life`: memory half-life $h_\beta$.
- `--readout-mu`: readout blend $\mu$.

During training, each optimizer group stores:

- `shrink`: the scheduled $\zeta_t$.
- `eta_unit`: $(1-\zeta_t)\rho$.
- `lr`: the additive step $\eta_t$.
- `step_scale`: the scheduled linear value used to derive $r_t$.
- `schedule_ratio`: the rate ratio $r_t$.

`--save-state-at decay-start` writes a full model, optimizer, and RNG state at
the start of the decay phase. Resuming that state with `--resume-state` lets the
tail schedule be changed without retraining the stable prefix.
