"""Edge/SOL splitting and N-region masking over a rho_pol grid.

Region-constant fitting (fit one set of operator params per region rather
than one per grid point) is the "regularized" fitting mode used throughout
the July 2026 presentation. This module isolates the masking logic so it can
be tested on its own and reused by both the fitter and scripts/scan_D_chi.py.
"""
from __future__ import annotations

import numpy as np


def edge_sol_mask(rho: np.ndarray, separatrix: float = 1.0) -> np.ndarray:
    """Boolean mask, True where rho is in the confined/edge region (rho < separatrix)."""
    return rho < separatrix


def region_edges(rho: np.ndarray, n_regions: int, separatrix: float = 1.0) -> np.ndarray:
    """Boundary values splitting [rho.min(), rho.max()] into n_regions regions.

    Odd n_regions: n_regions equal-width regions spanning the full range.
    Even n_regions: n_regions/2 equal-width regions inside the separatrix and
    n_regions/2 outside it, so the separatrix always falls on a region edge.
    """
    if n_regions < 1:
        raise ValueError("n_regions must be >= 1")

    if n_regions % 2 != 0:
        return np.linspace(rho.min(), rho.max(), n_regions + 1)

    n_half = n_regions // 2
    edges_in = np.linspace(rho.min(), separatrix, n_half + 1)
    edges_out = np.linspace(separatrix, rho.max(), n_half + 1)
    return np.concatenate((edges_in[:-1], edges_out))


def build_region_masks(rho: np.ndarray, n_regions: int, separatrix: float = 1.0) -> list[np.ndarray]:
    """Boolean masks partitioning `rho` into `n_regions` contiguous regions.

    If n_regions >= len(rho), falls back to point-by-point: one mask per grid
    point (each a length-1 True run), matching the "point-by-point" fitting
    mode from the original interactive tool.
    """
    n_points = len(rho)
    if n_regions >= n_points:
        return [np.arange(n_points) == i for i in range(n_points)]

    edges = region_edges(rho, n_regions, separatrix)
    masks = []
    n_edges = len(edges)
    for i in range(n_edges - 1):
        r_min, r_max = edges[i], edges[i + 1]
        if i == n_edges - 2:
            mask = (rho >= r_min) & (rho <= r_max)
        else:
            mask = (rho >= r_min) & (rho < r_max)
        if np.any(mask):
            masks.append(mask)
    return masks
