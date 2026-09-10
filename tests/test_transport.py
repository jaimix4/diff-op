import pytest

from diffop import operators
from diffop.transport import compute_flux, get_matrix_elements, matrix_elements_numeric

M_E = 9.109e-31
E_CHARGE = 1.602e-19


@pytest.mark.parametrize("name", ["gaussian", "sincx"])
def test_numeric_matches_closed_form(name):
    """The numeric-quadrature path (needed for future operators without a
    closed form) must agree with the closed-form matrix elements for the two
    operators that have both, or it can't be trusted for the ones that don't.
    """
    op = operators.get(name)
    A, alpha, n, T = 1.5, 1.2, 1e19, 100 * E_CHARGE

    analytic = op.matrix_elements(A, alpha, n, T)
    numeric = matrix_elements_numeric(op, A=A, alpha=alpha, m=M_E, T=T, n=n)

    for field in ["A11", "A12", "A21", "A22"]:
        a = getattr(analytic, field)
        b = getattr(numeric, field)
        assert a == pytest.approx(b, rel=1e-4)


@pytest.mark.parametrize("name", ["gaussian", "sincx"])
def test_get_matrix_elements_prefers_closed_form(name):
    op = operators.get(name)
    mat = get_matrix_elements(op, A=1.0, alpha=1.0, n=1e19, T=100 * E_CHARGE)
    assert mat == op.matrix_elements(1.0, 1.0, 1e19, 100 * E_CHARGE)


def test_compute_flux_zero_gradient_gives_zero_diffusive_flux():
    op = operators.get("sincx")
    result = compute_flux(op, A=1.0, alpha=1.0, n=1e19, T=100 * E_CHARGE, dn=0.0, dT=0.0, V=0.0)
    assert result.Gamma == 0.0
    assert result.q == 0.0


def test_compute_flux_convection_only():
    op = operators.get("sincx")
    result = compute_flux(op, A=1.0, alpha=1.0, n=1e19, T=100 * E_CHARGE, dn=0.0, dT=0.0, V=10.0)
    assert result.Gamma == pytest.approx(10.0 * 1e19)
    assert result.Gamma_C == result.Gamma
