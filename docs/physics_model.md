# Physics model: velocity-dependent diffusion-convection operator

Standing reference for the transport-matrix math this repo implements. Pulled from
the original README and the "Modeling turbulent transport with a velocity dependent
diffusion-convection operator" talk (Gkeyll team meeting, 8 July 2026). See code in
`src/diffop/` for the implementation; each section below names the module it lives in.

## 1. Motivation

Gkeyll's 2D2V edge/SOL model currently closes turbulent transport with a constant,
ad-hoc diffusion coefficient:

```
D_s(v) = D          (velocity-independent)
chi = 1.5 D          (fixed ratio, hardcoded)
```

The goal here is a velocity-dependent `D_s(v)` that is closer to first-principles
(thermal particles drive most turbulent transport; high-energy particles largely
don't), stays analytically or at least numerically tractable, and allows any
`chi/D` ratio rather than the fixed 1.5.

## 2. Extended diffusion-convection transport model

Rather than pure Fick's-law diffusion, the model keeps the off-diagonal
(thermodiffusion / Dufour) terms:

```
Gamma_s = -D11 grad(n_s) - D12 (n_s/T_s) grad(T_s) + V_s n_s
q_s     = -D21 T_s grad(n_s) - D22 n_s grad(T_s)          chi_s = D22
```

* **D11** — pure diffusion (density gradient -> particle flux)
* **D12** — thermodiffusion / Soret effect (temperature gradient -> particle flux;
  often drives inward pinches)
* **D21** — Dufour effect (density gradient -> heat flux)
* **D22** — thermal conductivity
* **V_s** — convective velocity (e.g. filament/blob transport, independent of
  local gradients)

## 3. Two-channel system & ambipolarity

Electrons and ions are fit as two coupled channels, each with their own operator
parameters, constrained by ambipolarity:

```
Gamma_e = Gamma_i = Gamma_total
```

This is enforced *implicitly* during fitting by fitting both `Gamma_e` and `Gamma_i`
against the same ground-truth `Gamma_data` (see `src/diffop/fitting.py`).

## 4. Building the transport matrix from D_s(v)

The matrix elements are velocity-space moments of a chosen scalar shape `D_s(v)`,
weighted by `J(v) = 4 pi v^2` and evaluated against a Maxwellian `f_M`:

```
Gamma_s = -Int J(v) D_s(v) grad(f_s) dv + Int J(v) V_s f_s dv       (0th moment)
q_s     = -Int (1/2 m_s v^2 - 3/2 T_s) J(v) D_s(v) grad(f_s) dv     (2nd moment)
```

with `f_s = f_M,s` (Maxwellian assumption) and speed-space integration
`v in [0, infinity)`. Expanding `grad(f_M) = (df_M/dn) grad(n) + (df_M/dT) grad(T)`
gives the four matrix elements as moment integrals:

```
A11 = Int J(v) D(v) df_M/dn dv          A12 = Int J(v) D(v) df_M/dT dv
A21 = Int w(v) J(v) D(v) df_M/dn dv     A22 = Int w(v) J(v) D(v) df_M/dT dv
```

where `w(v) = 1/2 m v^2 - 3/2 T`. This is implemented generically in
`src/diffop/transport.py`:

* If the operator shape has a **closed-form** solution to these integrals
  (`operator.matrix_elements`), that's used directly.
* Otherwise, `transport.matrix_elements_numeric` evaluates the four integrals by
  quadrature (`scipy.integrate.quad`) against the raw `operator.shape` — this is
  the path any future operator without an analytic solution needs. It's validated
  against the closed-form results in `tests/test_transport.py` (agreement to
  ~1e-6 relative or better for gaussian/sincx).

## 5. Candidate operator shapes

Implemented in `src/diffop/operators.py` as a registry — add a new shape by
writing `shape(v, A, alpha, m, T)` (+ optionally `matrix_elements(A, alpha, n, T)`
for a closed form) and calling `register(...)`; nothing else changes.

### gaussian

```
D_s(v) = A_s exp(-alpha_s m_s v^2 / 2T_s)

A11 = A / (1+alpha)^(3/2)
A12 = -3 A alpha n / (2 (1+alpha)^(5/2) T)
A21 = -3 A alpha T / (2 (1+alpha)^(5/2))
A22 = 3 A (2 + 3 alpha^2) n / (4 (1+alpha)^(7/2))

chi/D range: [0.9, 2.3]
```

### sincx

```
D_s(v) = A_s sinc(alpha_s sqrt(m_s v^2 / 2T_s)),   sinc(x) = sin(x)/x

A11 = A exp(-alpha^2/4)
A12 = -A exp(-alpha^2/4) alpha^2 n / (4T)
A21 = -A exp(-alpha^2/4) alpha^2 T / 4
A22 = A exp(-alpha^2/4) n (24 - 8 alpha^2 + alpha^4) / 16

chi/D range: [0.5, infinity)
```

Note the mass `m_s` cancels out of both closed forms entirely — only `alpha`, `n`,
`T` appear in the matrix elements. `m` is only needed for `shape(v, ...)` itself
(plotting/reference) and for the numeric-quadrature path used by operators without
a closed form.

## 6. Region masking (edge/SOL, N-region, point-by-point)

Operator parameters can be fit as one constant value everywhere, split into N
regions, or point-by-point. Implemented in `src/diffop/regions.py`:

* **Odd N**: N equal-width regions spanning the full rho range.
* **Even N**: N/2 equal-width regions inside the separatrix + N/2 outside, so the
  separatrix always falls exactly on a region boundary.
* **N >= number of grid points**: point-by-point (one region per grid point).

## 7. The optimization problem

Parameters `(A_e, alpha_e, V_e, A_i, alpha_i, V_i)` per region are fit against
ground-truth radial flux profiles via a normalized-MSE objective:

```
J = W1 ((Gamma_e - Gamma_data) / max|Gamma_data|)^2
  + W2 ((Gamma_i - Gamma_data) / max|Gamma_data|)^2
  + W3 ((q_e - q_e,data) / max|q_e,data|)^2
  + W4 ((q_i - q_i,data) / max|q_i,data|)^2
```

Normalization is necessary because particle flux (~1e20) and heat flux (~1e4)
live on wildly different scales — without it the optimizer would ignore the heat
flux entirely. The `W_k` let you trade off which flux the fit prioritizes.
Minimized per-region with `scipy.optimize.minimize(method="L-BFGS-B")` — see
`src/diffop/fitting.py`.

## 8. Ground-truth dataset

The current ground truth is AUG attached L-mode shot #36190 (GRILLIX, a global
drift-reduced Braginskii FCI code): digitized `n_e, T_e, T_i, D, chi_e, chi_i` vs
`rho_pol` from Zholobenko et al., NME 34 (2023) 101351, combined with a geqdsk
equilibrium (`data/raw/aug36190_mod.geqdsk`) for the `drho/dR` mapping factor.
See `scripts/extract_transport_data.py`.

## 9. Open questions / next steps (as of the 8 July 2026 talk)

* Near/after the separatrix, the diffusive model fits poorly and `Gamma` reverses
  sign around `rho_pol ~ 0.98` in the point-by-point fit — possibly indicates
  convective/filamentary transport the diffusive-only model can't capture
  (P. Manz et al., PoP 27 (2020) 022506).
* Eventually: derive the operator in field-aligned coordinates, discretize in DG,
  and run SOL/X-point Gkeyll simulations of AUG #36190 with different operators.
  This will need a numerical-integration path for operators without closed-form
  matrix elements — the `transport.matrix_elements_numeric` path above is built
  with that in mind.
