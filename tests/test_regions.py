import numpy as np
import pytest

from diffop.regions import build_region_masks, edge_sol_mask, region_edges


@pytest.fixture
def rho():
    return np.linspace(0.9, 1.05, 31)


def test_edge_sol_mask_splits_at_separatrix(rho):
    mask = edge_sol_mask(rho, separatrix=1.0)
    assert np.all(rho[mask] < 1.0)
    assert np.all(rho[~mask] >= 1.0)


@pytest.mark.parametrize("n_regions", [1, 2, 3, 4, 5, 6])
def test_masks_partition_all_points_exactly_once(rho, n_regions):
    masks = build_region_masks(rho, n_regions, separatrix=1.0)
    coverage = np.zeros(len(rho), dtype=int)
    for m in masks:
        coverage += m.astype(int)
    assert np.all(coverage == 1), "every point must belong to exactly one region"


def test_point_by_point_when_n_regions_exceeds_grid(rho):
    masks = build_region_masks(rho, n_regions=len(rho) + 10, separatrix=1.0)
    assert len(masks) == len(rho)
    for i, m in enumerate(masks):
        assert m.sum() == 1
        assert m[i]


def test_even_n_regions_splits_on_separatrix(rho):
    edges = region_edges(rho, n_regions=4, separatrix=1.0)
    assert 1.0 in edges


def test_odd_n_regions_ignores_separatrix(rho):
    edges = region_edges(rho, n_regions=3, separatrix=1.0)
    assert edges[0] == rho.min()
    assert edges[-1] == rho.max()
    assert len(edges) == 4
