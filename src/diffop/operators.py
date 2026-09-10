"""Registry of velocity-dependent diffusion operator shapes D_s(v).

Each operator is a candidate shape for D_s(v) in the transport model

    Gamma_s = -Int J(v) D_s(v) grad(f_s) dv + Int J(v) V_s f_s dv
    q_s     = -Int (1/2 m_s v^2 - 3/2 T_s) J(v) D_s(v) grad(f_s) dv

with J(v) = 4 pi v^2 and f_s assumed Maxwellian. An operator always provides
`shape(v, ...)` (the D_s(v) curve itself, for plotting/reference and for
numerical quadrature). If a closed-form solution for the moment integrals
above is known, it also provides `matrix_elements(...)` returning the
(A11, A12, A21, A22) transport matrix directly; otherwise transport.py falls
back to numerically integrating `shape` against the Maxwellian moments.

Add a new shape by writing `shape` (+ optionally `matrix_elements`) and
calling `register(...)` — nothing else in the package needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class MatrixElements:
    A11: float
    A12: float
    A21: float
    A22: float


@dataclass(frozen=True)
class Operator:
    name: str
    # D_s(v; A, alpha, m, T) -> array_like, same shape as v
    shape: Callable[..., np.ndarray]
    # (A, alpha, n, T) -> MatrixElements, closed form. None if not available.
    matrix_elements: Optional[Callable[..., MatrixElements]] = None
    # documented achievable chi/D range for this shape, if known (from the derivation)
    chi_over_D_range: Optional[tuple[float, float]] = None

    @property
    def has_closed_form(self) -> bool:
        return self.matrix_elements is not None


_REGISTRY: dict[str, Operator] = {}


def register(op: Operator) -> None:
    _REGISTRY[op.name] = op


def get(name: str) -> Operator:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown operator '{name}'. Available: {available()}") from None


def available() -> list[str]:
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# gaussian: D_s(v) = A exp(-alpha m v^2 / 2T)
# ---------------------------------------------------------------------------

def _gaussian_shape(v, A, alpha, m, T):
    return A * np.exp(-alpha * m * v**2 / (2 * T))


def _gaussian_matrix_elements(A, alpha, n, T) -> MatrixElements:
    A11 = A / (1 + alpha) ** 1.5
    A12 = -(3 * alpha * A * n) / (2 * (1 + alpha) ** 2.5 * T)
    A21 = -(3 * alpha * A * T) / (2 * (1 + alpha) ** 2.5)
    A22 = (3 * (2 + 3 * alpha**2) * A * n) / (4 * (1 + alpha) ** 3.5)
    return MatrixElements(A11, A12, A21, A22)


register(
    Operator(
        name="gaussian",
        shape=_gaussian_shape,
        matrix_elements=_gaussian_matrix_elements,
        chi_over_D_range=(0.9, 2.3),
    )
)


# ---------------------------------------------------------------------------
# sincx: D_s(v) = A sinc(alpha sqrt(m v^2 / 2T)),  sinc(x) = sin(x)/x
# ---------------------------------------------------------------------------

def _sincx_shape(v, A, alpha, m, T):
    x = alpha * np.sqrt(m * v**2 / (2 * T))
    return A * np.sinc(x / np.pi)  # np.sinc(y) = sin(pi y)/(pi y) -> sin(x)/x at y = x/pi


def _sincx_matrix_elements(A, alpha, n, T) -> MatrixElements:
    pref = A * np.exp(-alpha**2 / 4)
    A11 = pref
    A12 = -pref * (n * alpha**2) / (4 * T)
    A21 = -pref * (T * alpha**2) / 4
    A22 = pref * n * (24 - 8 * alpha**2 + alpha**4) / 16
    return MatrixElements(A11, A12, A21, A22)


register(
    Operator(
        name="sincx",
        shape=_sincx_shape,
        matrix_elements=_sincx_matrix_elements,
        chi_over_D_range=(0.5, np.inf),
    )
)
