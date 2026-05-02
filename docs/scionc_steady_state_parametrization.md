# ScionC Steady-State Parametrization

This note derives the ScionC training coordinates used for easier transfer
across batch size, block size, and gradient accumulation changes.

## One Group

For one optimizer group, let $X$ be the parameter tensor, $G$ the current
gradient, and $M$ the EMA memory. One optimizer update advances the training
count by $\Delta\tau$ processed tokens:

```math
\Delta\tau
=
\text{batch size}
\cdot
\text{block size}
\cdot
\text{gradient accumulation}.
```

Use half-life coordinates for memory and direct shrinkage:

```math
\beta = 2^{-\Delta\tau/h_\beta},
\qquad
\zeta = 2^{-\Delta\tau/h_\zeta}.
```

The memory update and readout are:

```math
\bar M = \beta M + (1-\beta)G,
\qquad
R = (1-\mu)G + \mu\bar M.
```

The ULMO returns a unit action in the group geometry:

```math
V = \operatorname{ULMO}(R).
```

The parameter update is:

```math
X_{\mathrm{new}} = \zeta X + \eta_t V.
```

Here $\zeta$ is chosen from the independent shrink half-life. The additive step
$\eta_t$ is derived from the target radius and the dimensionless schedule.

## Corrected RMS Step

The corrected steady-state calculation balances second moments, not the
worst-case triangle inequality. Assume the update atoms have effective squared
size:

```math
\mathbb{E}\|V\|^2 = C.
```

For Lion-K style normalized updates, write:

```math
C = \frac{c_u^2 S}{q}.
```

Here $c_u^2$ is the atom squared-norm scale, $S$ is the momentum amplification
factor, and $q$ is the cautious-mask keep fraction. The second-moment steady
state at radius $\rho$ satisfies:

```math
\rho^2
=
\zeta^2\rho^2
+
\eta^2 C.
```

Thus:

```math
1-\zeta^2
=
\frac{\eta^2 c_u^2 S}{q\rho^2}.
```

Solving for the additive step gives:

```math
\eta_t
=
s_t\rho
\sqrt{
\frac{q(1-\zeta^2)}{c_u^2 S}
}.
```

This is the active ScionC parametrization. In the current training script,
ScionC has no cautious masking and uses unit ULMO atoms, so $q=1$ and $c_u^2=1$:

```math
\eta_t
=
s_t\rho
\sqrt{
\frac{1-\zeta^2}{S}
}.
```

The momentum amplification factor is computed from the readout blend $\mu$ and
memory retention $\beta$:

```math
S
=
\frac{1+\beta}
{(1-\mu\beta)^2(1+\beta)+(\mu\beta)^2(1-\beta)}.
```

## Transfer Rule

When the count increment changes from $\Delta\tau$ to $\Delta\tau_\star$, keep
the semantic hyperparameters fixed:

```math
\rho_\star = \rho,
\qquad
h_{\zeta,\star} = h_\zeta,
\qquad
s_{t,\star} = s_t,
\qquad
h_{\beta,\star} = h_\beta,
\qquad
\mu_\star = \mu.
```

Then recompute the raw factors from the new count increment:

```math
\zeta_\star = 2^{-\Delta\tau_\star/h_\zeta},
\qquad
\beta_\star = 2^{-\Delta\tau_\star/h_\beta},
```

```math
\eta_{t,\star}
=
s_t\rho
\sqrt{
\frac{1-\zeta_\star^2}{S_\star}
}.
```

This keeps shrink independent of the additive step schedule while preserving
the memory half-life, shrink half-life, steady-state radius, and dimensionless
action schedule in token-count units.

## Relation To Corrected Decay

The lower-level corrected-decay approximation solves the same RMS balance in
the small-shrink regime. Let $d=1-\zeta$. Since
$1-\zeta^2=(1-\zeta)(1+\zeta)\approx 2d$:

```math
d
\approx
\frac{\eta_t^2 c_u^2 S}{2q\rho^2}.
```

This is the existing corrected decoupled-decay formula, but used in the forward
direction: choose independent shrink first, then derive the additive step. The
recipe does not set shrink from the additive-step schedule.

## Hard Invariant Bound

A more conservative alternative is to require the whole radius ball to be
invariant by the triangle inequality. If $\|V\|\le 1$, then:

```math
\|X_{\mathrm{new}}\|
\le
\zeta\|X\|+\eta_t.
```

The ball $\|X\|\le\rho$ is invariant when:

```math
\eta_t
\le
(1-\zeta)\rho.
```

At equality:

```math
X_{\mathrm{new}}
=
\zeta X + (1-\zeta)\rho V.
```

This bound is useful for comparison, but it is not the active ScionC step-size
correction.

## User-Facing Coordinates

The primary transfer coordinates are:

```math
\rho,
\qquad
h_\zeta,
\qquad
s_t,
\qquad
h_\beta,
\qquad
\mu.
```

The optimizer may store the derived additive step in an internal field named
`lr` for PyTorch compatibility, but conceptually that field is $\eta_t$.
