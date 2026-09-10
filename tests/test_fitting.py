import numpy as np

from diffop.fitting import fit_two_channel

E_CHARGE = 1.602e-19


def _synthetic_profile(n_points=6):
    rho = np.linspace(0.9, 1.1, n_points)
    n = np.full(n_points, 1e19)
    Te = np.full(n_points, 150.0)
    Ti = np.full(n_points, 180.0)
    return rho, n, Te, Ti


def test_fit_two_channel_output_shapes_and_finite():
    rho, n, Te, Ti = _synthetic_profile()
    dn = np.linspace(-1e20, -0.5e20, len(rho))
    dTe = np.linspace(-3000.0, -1500.0, len(rho))
    dTi = np.linspace(-3500.0, -1800.0, len(rho))
    Gamma_true = np.linspace(1e20, 0.6e20, len(rho))
    qe_true = np.linspace(5000.0, 2500.0, len(rho))
    qi_true = np.linspace(6000.0, 3000.0, len(rho))

    result = fit_two_channel(
        rho, n, Te, Ti, dn, dTe, dTi, Gamma_true, qe_true, qi_true,
        operator="sincx", n_regions=1, use_convection=False,
    )

    for channel in (result.electron, result.ion):
        assert channel.A.shape == rho.shape
        assert np.all(np.isfinite(channel.Gamma))
        assert np.all(np.isfinite(channel.q))


def test_zero_gradient_and_no_convection_forces_zero_flux():
    """With grad(n)=grad(T)=0 everywhere and convection disabled, Gamma and q
    must be exactly zero regardless of what (A, alpha) the optimizer picks --
    a deterministic check that doesn't depend on the optimizer actually
    converging well.
    """
    rho, n, Te, Ti = _synthetic_profile()
    zeros = np.zeros(len(rho))

    result = fit_two_channel(
        rho, n, Te, Ti, zeros, zeros, zeros, zeros, zeros, zeros,
        operator="gaussian", n_regions=1, use_convection=False,
    )

    for channel in (result.electron, result.ion):
        assert np.allclose(channel.Gamma, 0.0)
        assert np.allclose(channel.q, 0.0)
        assert np.allclose(channel.V, 0.0)


def test_point_by_point_matches_number_of_grid_points():
    rho, n, Te, Ti = _synthetic_profile(n_points=5)
    ones = np.ones(len(rho))
    result = fit_two_channel(
        rho, n, Te, Ti, -ones, -ones, -ones, ones, ones, ones,
        operator="sincx", n_regions=len(rho), use_convection=False,
    )
    assert len(result.masks) == len(rho)
