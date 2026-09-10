# CLAUDE.md

Context for working in this repo. Full physics derivation: `docs/physics_model.md`.

## What this is

PhD research repo (DIFFER) developing reduced turbulent-transport models for the
edge/SOL of 2D gyrokinetic Gkeyll simulations (2D2V, DG discretization). Gkeyll
currently closes turbulent transport with a constant, ad-hoc diffusion coefficient
(`D_s(v) = D`, `chi = 1.5D`, ratio hardcoded). The goal is a velocity-dependent
`D_s(v)` closer to first-principles, analytically or numerically tractable, that
allows any `chi/D` ratio.

Presented at the Gkeyll team meeting, 8 July 2026 ("Modeling turbulent transport
with a velocity dependent diffusion-convection operator").

## Physics model (see docs/physics_model.md for full derivation)

Off-diagonal diffusion-convection transport matrix (includes thermodiffusion/Dufour):

```
Gamma_s = -D11 grad(n_s) - D12 (n_s/T_s) grad(T_s) + V_s n_s
q_s     = -D21 T_s grad(n_s) - D22 n_s grad(T_s)          chi_s = D22
```

built from velocity moments of a chosen scalar shape `D_s(v)`:

```
Gamma_s = -Int J(v) D_s(v) grad(f_s) dv + Int J(v) V_s f_s dv     (0th moment)
q_s     = -Int (1/2 m_s v^2 - 3/2 T_s) J(v) D_s(v) grad(f_s) dv   (2nd moment)
```

with `J(v) = 4 pi v^2`, `f_s` assumed Maxwellian. Electron/ion channels are fit
jointly, coupled via ambipolarity (`Gamma_e = Gamma_i`), enforced implicitly by
fitting both channels against the same ground-truth `Gamma_data`.

Two operator shapes currently implemented, both with closed-form matrix elements:
`gaussian` (`chi/D` in [0.9, 2.3]) and `sincx` (`chi/D` in [0.5, infinity)). Full
formulas in `docs/physics_model.md` section 5 and `src/diffop/operators.py`.

## Repo layout

```
src/diffop/
    operators.py   D_s(v) shape registry (register a new shape via register(Operator(...)),
                    nothing else needs to change). Each op provides `shape(v,...)`
                    always, and `matrix_elements(...)` only if a closed form exists.
    transport.py    Gamma/q from an operator's matrix elements + convection.
                    Falls back to numeric quadrature (matrix_elements_numeric)
                    for operators without a closed form -- validated against
                    the closed-form gaussian/sincx results in tests/.
    regions.py      edge/SOL split + N-region masking (odd N = N equal regions,
                    even N = N/2 core + N/2 SOL, N >= grid size = point-by-point).
    fitting.py      objective function + L-BFGS-B optimizer, joint electron/ion
                    fit per region (fit_two_channel).
scripts/            thin entry points calling src/diffop:
    extract_transport_data.py   AUG #36190 (GRILLIX) data pipeline -> data/processed/
    run_interactive_fit.py      matplotlib slider UI for interactive fitting
    (scan_D_chi.py — planned, not yet built: D0/alpha -> D/chi region-config scan)
data/raw/           geqdsk equilibria, raw simulation exports
data/processed/     generated CSVs (aug36190_master_dataset_2.csv)
docs/physics_model.md   standing physics/math reference
results/            output plots from scripts/
tests/              unit tests for src/diffop
legacy/presentation_2026_07_08/
                    FROZEN, self-contained copies of every script + data + PNG
                    that produced the 8 July 2026 slides. Never modify -- exists
                    purely so those results stay reproducible. Everything in
                    src/diffop is a from-scratch generalization of the logic
                    that used to live inline in
                    legacy/presentation_2026_07_08/5_two_channel_fit.py and
                    transport_data_36190_zhobo.py, not a copy of it.
```

## Environment

```bash
conda env create -f environment.yml   # creates env "diffop", installs src/diffop editable
conda activate diffop
python -m pytest tests/
```

See the environment note in README.md — this env is provisional and may get
consolidated into one shared env for this repo + the Gkeyll source tree later.

## Conventions worth knowing

* Temperatures are in eV in the data/plotting layer but must be Joules (`T * E_CHARGE`)
  wherever they enter the Maxwellian/moment-integral math in `transport.py` --
  `E_CHARGE = 1.602e-19` is defined there.
* The mass `m_s` cancels out of the gaussian/sincx closed-form matrix elements
  entirely (only `alpha`, `n`, `T` appear) -- it's only needed for `operator.shape(v,...)`
  itself (plotting/reference) and the numeric-quadrature fallback.
* `fit_two_channel`'s per-region objective enforces ambipolarity implicitly: both
  the electron and ion channel are fit against the *same* `Gamma_true` array in one
  joint `minimize()` call (6 params: `A_e, alpha_e, V_e, A_i, alpha_i, V_i`).
* Region-constant fits can be highly degenerate -- very different `(A, alpha)`
  pairs can give nearly identical output fluxes (confirmed empirically during the
  restructure: refit from the same initial guess and bounds converged to different
  parameter values but near-identical flux curves). Don't over-interpret a single
  fitted `(A, alpha)` pair physically without checking the objective landscape.

## What's next (don't build ahead of this, but don't block it either)

* `scripts/scan_D_chi.py` -- scan D0/alpha (and resulting `chi/D`) under different
  region configurations (constant / edge-SOL / N-subdivided) against multiple
  profiles (AUG #36190 + synthetic linear/tanh profiles). Not built yet.
* More `D_s(v)` shapes beyond gaussian/sincx -- add via `operators.register(...)`.
* Other simulation datasets beyond AUG #36190/GRILLIX.
* Eventually: derive the operator in field-aligned coordinates, discretize in DG,
  and run SOL/X-point Gkeyll simulations with different operators. Will need the
  numeric-integration path in `transport.py` for operators without closed forms,
  and will connect this repo to an actual Gkeyll source checkout (a separate
  folder in the workspace, not yet added).
