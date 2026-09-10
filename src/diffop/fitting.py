"""Fit operator parameters to ground-truth flux profiles.

Generalizes the objective function + L-BFGS-B optimizer from the original
interactive fitter (legacy/presentation_2026_07_08/5_two_channel_fit.py),
decoupled from any one analytic operator: pass any registered operator name
(or an Operator instance directly) and it drives transport.compute_flux_array
instead of an embedded if/elif model.

Electron and ion channels are fit jointly per region, both against the same
Gamma_true, which is what enforces the ambipolarity constraint Gamma_e = Gamma_i.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from . import operators as operators_module
from .operators import Operator
from .regions import build_region_masks
from .transport import compute_flux_array


@dataclass
class ChannelResult:
    A: np.ndarray
    alpha: np.ndarray
    V: np.ndarray
    Gamma: np.ndarray
    Gamma_n: np.ndarray
    Gamma_T: np.ndarray
    Gamma_C: np.ndarray
    q: np.ndarray
    q_n: np.ndarray
    q_T: np.ndarray


@dataclass
class TwoChannelResult:
    electron: ChannelResult
    ion: ChannelResult
    masks: list


def _resolve_operator(operator) -> Operator:
    return operators_module.get(operator) if isinstance(operator, str) else operator


def _region_objective(params, n, Te, Ti, dn, dTe, dTi, G_true, qe_true, qi_true, operator, w1, w2, w3, w4, m_e, m_i):
    A_e, alpha_e, C_e, A_i, alpha_i, C_i = params

    fe = compute_flux_array(operator, A=A_e, alpha=alpha_e, n=n, T=Te, dn=dn, dT=dTe, V=C_e, m=m_e)
    fi = compute_flux_array(operator, A=A_i, alpha=alpha_i, n=n, T=Ti, dn=dn, dT=dTi, V=C_i, m=m_i)

    norm_G = np.max(np.abs(G_true)) + 1e-30
    norm_qe = np.max(np.abs(qe_true)) + 1e-30
    norm_qi = np.max(np.abs(qi_true)) + 1e-30

    err_Ge = np.mean(((G_true - fe.Gamma) / norm_G) ** 2)
    err_Gi = np.mean(((G_true - fi.Gamma) / norm_G) ** 2)
    err_qe = np.mean(((qe_true - fe.q) / norm_qe) ** 2)
    err_qi = np.mean(((qi_true - fi.q) / norm_qi) ** 2)

    return w1 * err_Ge + w2 * err_Gi + w3 * err_qe + w4 * err_qi


def fit_two_channel(
    rho,
    n,
    Te,
    Ti,
    dn,
    dTe,
    dTi,
    Gamma_true,
    qe_true,
    qi_true,
    *,
    operator="sincx",
    n_regions=1,
    separatrix=1.0,
    weights=(1.0, 1.0, 1.0, 1.0),
    use_convection=False,
    conv_range=None,
    alpha_bounds=(0.0, 20.0),
    m_e=None,
    m_i=None,
) -> TwoChannelResult:
    """Fit (A, alpha, [convection]) per region for electron + ion channels jointly.

    n_regions=1 -> single constant value everywhere. Even n_regions split
    n_regions/2 core + n_regions/2 SOL around `separatrix` (see regions.py).
    n_regions >= len(rho) -> point-by-point.
    """
    op = _resolve_operator(operator)
    w1, w2, w3, w4 = weights
    masks = build_region_masks(np.asarray(rho), n_regions, separatrix)

    if conv_range is None:
        conv_range = (np.min(rho), np.max(rho))
    rho_c_min, rho_c_max = conv_range

    n, Te, Ti, dn, dTe, dTi, Gamma_true, qe_true, qi_true = (
        np.asarray(x, dtype=float) for x in (n, Te, Ti, dn, dTe, dTi, Gamma_true, qe_true, qi_true)
    )
    rho = np.asarray(rho, dtype=float)

    out_e = {k: np.zeros_like(rho) for k in ("A", "alpha", "V")}
    out_i = {k: np.zeros_like(rho) for k in ("A", "alpha", "V")}

    initial_guess = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    for mask in masks:
        r_mid = np.mean(rho[mask])
        c_bnd = (None, None) if (use_convection and rho_c_min <= r_mid <= rho_c_max) else (0.0, 0.0)
        bounds = ((None, None), alpha_bounds, c_bnd, (None, None), alpha_bounds, c_bnd)

        args = (
            n[mask], Te[mask], Ti[mask], dn[mask], dTe[mask], dTi[mask],
            Gamma_true[mask], qe_true[mask], qi_true[mask],
            op, w1, w2, w3, w4, m_e, m_i,
        )
        res = minimize(_region_objective, initial_guess, args=args, bounds=bounds, method="L-BFGS-B")
        A_e, alpha_e, C_e, A_i, alpha_i, C_i = res.x

        out_e["A"][mask], out_e["alpha"][mask], out_e["V"][mask] = A_e, alpha_e, C_e
        out_i["A"][mask], out_i["alpha"][mask], out_i["V"][mask] = A_i, alpha_i, C_i

    fe = compute_flux_array(op, A=out_e["A"], alpha=out_e["alpha"], n=n, T=Te, dn=dn, dT=dTe, V=out_e["V"], m=m_e)
    fi = compute_flux_array(op, A=out_i["A"], alpha=out_i["alpha"], n=n, T=Ti, dn=dn, dT=dTi, V=out_i["V"], m=m_i)

    electron = ChannelResult(out_e["A"], out_e["alpha"], out_e["V"], fe.Gamma, fe.Gamma_n, fe.Gamma_T, fe.Gamma_C, fe.q, fe.q_n, fe.q_T)
    ion = ChannelResult(out_i["A"], out_i["alpha"], out_i["V"], fi.Gamma, fi.Gamma_n, fi.Gamma_T, fi.Gamma_C, fi.q, fi.q_n, fi.q_T)
    return TwoChannelResult(electron=electron, ion=ion, masks=masks)
