# diff-op

Velocity-dependent diffusion-convection transport models for the edge/SOL of 2D
gyrokinetic Gkeyll simulations (2D2V, DG discretization). PhD research project at
DIFFER. See [`docs/physics_model.md`](docs/physics_model.md) for the full physics
derivation and math reference.

## Layout

```
src/diffop/     the package: operators.py (D_s(v) shape registry), transport.py
                (Gamma/q from an operator's matrix elements, closed-form or
                numeric quadrature), regions.py (edge/SOL + N-region masking),
                fitting.py (objective + L-BFGS-B optimizer)
scripts/        thin entry points that call src/diffop
data/raw/       geqdsk equilibria, raw simulation exports
data/processed/ generated CSVs (e.g. aug36190_master_dataset_2.csv)
docs/           physics_model.md — the transport-matrix math as a standing reference
results/        output plots from scripts/
tests/          unit tests for src/diffop
legacy/presentation_2026_07_08/
                frozen, self-contained copies of every script (+ data + PNGs)
                that produced the 8 July 2026 Gkeyll team meeting slides. Not
                touched going forward — exists purely so those results stay
                reproducible without depending on src/diffop's evolution.
```

## Environment

```bash
conda env create -f environment.yml
conda activate diffop
```

This installs `src/diffop` itself in editable mode (`pip install -e .`), so
`import diffop` works from anywhere once the env is active.

> Note: this is a fresh env created for this restructuring pass. There's already
> another env this project has used, and the plan is likely to consolidate into
> one shared environment for everything touching this repo + the Gkeyll source
> tree once that phase starts — treat `environment.yml` here as provisional
> until that consolidation happens.

## Physics summary

Reduced transport model with off-diagonal (thermodiffusion/Dufour) terms:

```
Gamma_s = -D11 grad(n_s) - D12 (n_s/T_s) grad(T_s) + V_s n_s
q_s     = -D21 T_s grad(n_s) - D22 n_s grad(T_s)          chi_s = D22
```

built from velocity moments of a chosen scalar shape `D_s(v)` (currently
`gaussian` and `sincx`, see `src/diffop/operators.py`), fit to ground-truth radial
flux profiles from a GRILLIX simulation of AUG shot #36190 via a normalized-MSE
objective. Full derivation, both operators' closed forms, and the fitting
objective: [`docs/physics_model.md`](docs/physics_model.md).
