# 4-Moment Spectral Bound

In `scionc/ulmos/core.py`, the ULMO optimizer requires a bound on the spectral norm (the maximum eigenvalue) of a matrix $X$ to safely compute the Gram-Newton-Schulz iteration. Given the Gram matrix $G = X X^T$, let its $n$ non-negative eigenvalues be $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n \ge 0$.

Instead of computing the exact maximum eigenvalue $\lambda_1$, we bound it using the first four moments of the eigenvalue distribution. We define the normalized sum of the $k$-th powers of the eigenvalues:

```math
t_1 = \sum_{i=1}^n \lambda_i = \operatorname{tr}(G)
```
```math
m_k = \frac{\sum_{i=1}^n \lambda_i^k}{\left(\sum_{i=1}^n \lambda_i\right)^k} = \frac{\operatorname{tr}(G^k)}{\operatorname{tr}(G)^k}, \quad \text{for } k=2,3,4.
```

By definition, we also have $m_1 = 1$. Let $p_i = \lambda_i / t_1$ be the relative eigenvalues. They satisfy $p_i \ge 0$, $\sum_{i=1}^n p_i = 1$, and $\sum_{i=1}^n p_i^k = m_k$. 
Our goal is to find the maximum possible value of the largest relative eigenvalue, $\beta = p_1$, subject to these 4 moment constraints.

## The Truncated Moment Problem

If we assume one eigenvalue takes the value $t \in [0, 1]$, the remaining $n-1$ eigenvalues must account for the remaining moment mass. For the remaining points $p_2, \dots, p_n$, their power sums are:

```math
\begin{aligned}
s_0 &= n - 1 \\
s_1 &= 1 - t \\
s_2 &= m_2 - t^2 \\
s_3 &= m_3 - t^3 \\
s_4 &= m_4 - t^4
\end{aligned}
```

For these $s_k$ to represent a valid set of real points, they must form positive semi-definite Hankel matrices. The relevant Hankel matrices are:

```math
H_2 = \begin{pmatrix} s_2 & s_3 \\ s_3 & s_4 \end{pmatrix}, \qquad
M = \begin{pmatrix} s_0 & s_1 & s_2 \\ s_1 & s_2 & s_3 \\ s_2 & s_3 & s_4 \end{pmatrix}
```

The feasibility of $t$ requires that $\det(H_2) \ge 0$ and $\det(M) \ge 0$. 

## Hankel Determinant Constraints

The first constraint $\det(H_2) \ge 0$ gives:
```math
e_0(t) = s_2 s_4 - s_3^2 = (m_2 - t^2)(m_4 - t^4) - (m_3 - t^3)^2 \ge 0
```

The second constraint $\det(M) \ge 0$ expands into a degree-4 polynomial in $t$. By substituting the $s_k$ definitions and simplifying, we get:
```math
\det(M)(t) = d_4 t^4 + d_3 t^3 + d_2 t^2 + d_1 t + d_0 \ge 0
```
where the coefficients are independent of $t$:
```math
\begin{aligned}
d_4 &= 1 - n m_2 \\
d_3 &= 2(n m_3 - m_2) \\
d_2 &= 3 m_2^2 - 2 m_3 - n m_4 \\
d_1 &= 2(m_4 - m_2 m_3) \\
d_0 &= (n - 1)(m_2 m_4 - m_3^2) - m_2^3 + 2 m_2 m_3 - m_4
\end{aligned}
```

## Bisection for the Spectral Bound

To find the optimal upper bound $\beta$, the `_moment4_beta_interval` function performs a bisection search over $t \in [0, 1]$. 

It initializes the search interval using weaker bounds (such as the 2-moment variance bound $m_2$ and the Cauchy-Schwarz $m_4$ bounds). In each bisection step, it checks if the candidate $t$ is feasible by verifying:
```math
\min(e_0(t), \det(M)(t)) \ge - \epsilon_{\text{tol}}
```

The maximum feasible $t$ is guaranteed to upper-bound the true spectral norm $\lambda_1 / t_1$, providing a strict, safety-guaranteed step scale for the Gram-Newton-Schulz iteration while using only inexpensive trace operations.
