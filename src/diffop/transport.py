"""Build particle/heat fluxes (Gamma, q) from an operator's D_s(v) + convection.

Physics recap (see docs/physics_model.md for the full derivation):

    Gamma_s = -A11 grad(n) - A12 grad(T) + V_s n_s
    q_s     = -A21 grad(n) - A22 grad(T)          (chi_s = A22 / n_s convention
                                                    used in the fitter is folded
                                                    into A22 directly here)

(A11, A12, A21, A22) are velocity-space moments of a chosen D_s(v):

    A11 = Int J(v) D_s(v) df_M/dn dv
    A12 = Int J(v) D_s(v) df_M/dT dv
    A21 = Int (1/2 m v^2 - 3/2 T) J(v) D_s(v) df_M/dn dv
    A22 = Int (1/2 m v^2 - 3/2 T) J(v) D_s(v) df_M/dT dv

with J(v) = 4 pi v^2 and f_M the Maxwellian. If `operator.matrix_elements` is
available (closed form), it is used directly. Otherwise these four integrals
are evaluated numerically by quadrature against `operator.shape` — this is
the path any future operator without a closed-form solution needs, and it is
validated against the closed-form gaussian/sincx results in tests/.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate

from .operators import MatrixElements, Operator

E_CHARGE = 1.602e-19  # J/eV, matches the eV-based T convention used throughout


@dataclass(frozen=True)
class FluxResult:
    Gamma: float
    q: float
    Gamma_n: float   # density-gradient (diffusive) contribution to Gamma
    Gamma_T: float   # temperature-gradient (thermodiffusion) contribution to Gamma
    Gamma_C: float   # convective contribution to Gamma
    q_n: float       # density-gradient (Dufour) contribution to q
    q_T: float       # temperature-gradient (conductive) contribution to q
    matrix: MatrixElements


def _maxwellian_speed_pdf(v, m, T):
    """f_M(v) per unit density: n (m / 2 pi T)^1.5 exp(-m v^2 / 2T), n factored out."""
    return (m / (2 * np.pi * T)) ** 1.5 * np.exp(-m * v**2 / (2 * T))


def matrix_elements_numeric(operator: Operator, *, A, alpha, m, T, n, v_max_factor=15.0) -> MatrixElements:
    """Numerically integrate the four moment integrals for an operator lacking
    a closed form. Works for any operator (closed-form ones can use this too
    as a cross-check — see tests/test_transport.py).
    """
    vth = np.sqrt(2 * T / m)
    v_max = v_max_factor * vth

    def J(v):
        return 4 * np.pi * v**2

    def f_dn(v):
        # d f_M / dn = f_M / n, i.e. the per-unit-density Maxwellian
        return _maxwellian_speed_pdf(v, m, T)

    def f_dT(v):
        # d f_M / dT = n * fM_per_n/T * (m v^2 / 2T - 3/2) -- the n factor is
        # applied where this is used below (A12, A22), not A11/A21.
        fM_per_n = _maxwellian_speed_pdf(v, m, T)
        return fM_per_n / T * (m * v**2 / (2 * T) - 1.5)

    def D(v):
        return operator.shape(v, A, alpha, m, T)

    def weight(v):
        return 0.5 * m * v**2 - 1.5 * T

    A11, _ = integrate.quad(lambda v: J(v) * D(v) * f_dn(v), 0, v_max)
    A12, _ = integrate.quad(lambda v: J(v) * D(v) * f_dT(v), 0, v_max)
    A21, _ = integrate.quad(lambda v: weight(v) * J(v) * D(v) * f_dn(v), 0, v_max)
    A22, _ = integrate.quad(lambda v: weight(v) * J(v) * D(v) * f_dT(v), 0, v_max)

    # A11, A21 come from d f_M/dn (no explicit n factor); A12, A22 come from
    # d f_M/dT = n * (...), so the n factor is applied here.
    return MatrixElements(A11, A12 * n, A21, A22 * n)


def get_matrix_elements(operator: Operator, *, A, alpha, n, T, m=None) -> MatrixElements:
    """Closed form if the operator has one, else numeric quadrature (requires m)."""
    if operator.has_closed_form:
        return operator.matrix_elements(A, alpha, n, T)
    if m is None:
        raise ValueError(f"operator '{operator.name}' has no closed form; pass m for numeric integration")
    return matrix_elements_numeric(operator, A=A, alpha=alpha, m=m, T=T, n=n)


def compute_flux(
    operator: Operator,
    *,
    A: float,
    alpha: float,
    n: float,
    T: float,
    dn: float,
    dT: float,
    V: float = 0.0,
    m: float | None = None,
    heat_in_joules: bool = True,
) -> FluxResult:
    """Gamma, q for one species/channel from operator params + local plasma state.

    `dn`, `dT` are the (already computed) radial gradients grad(n), grad(T).
    `heat_in_joules` multiplies q by E_CHARGE, matching the eV-temperature,
    Joule-heat-flux convention used in the AUG #36190 dataset.
    """
    mat = get_matrix_elements(operator, A=A, alpha=alpha, n=n, T=T, m=m)

    Gamma_n = -mat.A11 * dn
    Gamma_T = -mat.A12 * dT
    Gamma_C = V * n
    Gamma = Gamma_n + Gamma_T + Gamma_C

    unit = E_CHARGE if heat_in_joules else 1.0
    q_n = -mat.A21 * dn * unit
    q_T = -mat.A22 * dT * unit
    q = q_n + q_T

    return FluxResult(Gamma, q, Gamma_n, Gamma_T, Gamma_C, q_n, q_T, mat)


def compute_flux_array(operator: Operator, *, A, alpha, n, T, dn, dT, V=0.0, m=None, heat_in_joules=True) -> FluxResult:
    """Vectorized version of compute_flux over arrays of (n, T, dn, dT, [A, alpha, V]).

    A, alpha, V may be scalars (constant model) or arrays broadcastable with
    n/T/dn/dT (regional/point-by-point models).
    """
    n, T, dn, dT, A, alpha, V = np.broadcast_arrays(n, T, dn, dT, A, alpha, V)

    if operator.has_closed_form:
        mat = operator.matrix_elements(A, alpha, n, T)
    else:
        if m is None:
            raise ValueError(f"operator '{operator.name}' has no closed form; pass m for numeric integration")
        rows = [
            matrix_elements_numeric(operator, A=Ai, alpha=ai, m=m, T=Ti, n=ni)
            for Ai, ai, ni, Ti in zip(np.atleast_1d(A), np.atleast_1d(alpha), np.atleast_1d(n), np.atleast_1d(T))
        ]
        mat = MatrixElements(
            A11=np.array([r.A11 for r in rows]),
            A12=np.array([r.A12 for r in rows]),
            A21=np.array([r.A21 for r in rows]),
            A22=np.array([r.A22 for r in rows]),
        )

    Gamma_n = -mat.A11 * dn
    Gamma_T = -mat.A12 * dT
    Gamma_C = V * n
    Gamma = Gamma_n + Gamma_T + Gamma_C

    unit = E_CHARGE if heat_in_joules else 1.0
    q_n = -mat.A21 * dn * unit
    q_T = -mat.A22 * dT * unit
    q = q_n + q_T

    return FluxResult(Gamma, q, Gamma_n, Gamma_T, Gamma_C, q_n, q_T, mat)
