import numpy as np

from diffop import operators


def test_registry_has_expected_operators():
    assert set(operators.available()) >= {"gaussian", "sincx"}


def test_gaussian_and_sincx_have_closed_forms():
    for name in ["gaussian", "sincx"]:
        op = operators.get(name)
        assert op.has_closed_form
        assert op.chi_over_D_range is not None


def test_unknown_operator_raises():
    try:
        operators.get("does_not_exist")
    except KeyError as e:
        assert "does_not_exist" in str(e)
    else:
        raise AssertionError("expected KeyError for unknown operator")


def test_gaussian_shape_peaks_at_zero_velocity():
    op = operators.get("gaussian")
    v = np.array([0.0, 1e5, 1e6])
    D = op.shape(v, A=1.0, alpha=1.0, m=9.109e-31, T=100 * 1.602e-19)
    assert D[0] == max(D)
