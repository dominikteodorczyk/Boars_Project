import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from model.utils.utils import get_geom_centroid
from model.src.gravity import (
    exponential_deterrence,
    power_law_deterrence,
    compute_distance_matrix,
    Gravity,
)


def test_exponential_deterrence():
    """Exponential deterrence should equal exp(-rate * d)."""
    distances = np.array([0.0, 1.0, 2.0])
    rate = 0.5
    expected = np.exp(-rate * distances)
    result = exponential_deterrence(distances, rate)
    assert np.allclose(result, expected)


def test_power_law_deterrence_with_zero_handling():
    """Power‑law deterrence should handle zeros safely and match expected values."""
    distances = np.array([0.0, 1.0, 2.0])
    exponent = -2.0
    result = power_law_deterrence(distances, exponent)

    # The zero distance should have been replaced with a small epsilon and remain finite > 0
    assert result[0] > 0.0 and np.isfinite(result[0])

    # Non‑zero values should match d ** exponent
    expected_non_zero = np.power(distances[1:], exponent)
    assert np.allclose(result[1:], expected_non_zero)


def test_compute_distance_matrix():
    """Distance matrix should compute Euclidean distances between centroids correctly."""
    # Two points 5 units apart (3‑4‑5 triangle)
    gdf = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(3, 4)],
        crs="EPSG:4326",
    )

    # Identity centroid function for Points (x, y)
    dist_matrix = compute_distance_matrix(gdf, origins=[0, 1], centroid_func=get_geom_centroid)

    expected = np.array([[0.0, 5.0], [5.0, 0.0]])
    assert np.allclose(dist_matrix, expected)


def test_gravity_compute_flows():
    """Gravity model should produce expected flow matrix for a simple 2‑cell system."""
    gravity = Gravity(deterrence_type="power_law", deterrence_params=[-2.0])
    dist_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
    origin_relevances = np.array([1.0, 2.0])
    dest_relevances = np.array([3.0, 4.0])

    flows = gravity._compute_gravity_score(0, dist_matrix, origin_relevances, dest_relevances)

    expected = np.array(
        [
            [0.0, 1.0 ** -2.0 * 4.0 * 1.0],
            [1.0 ** -2.0 * 3.0 * 2.0, 0.0],
        ]
    )
    assert np.allclose(flows, expected)


def test_gravity_singly_constrained_vector():
    """Singly-constrained case: test if self-flow is zeroed correctly when origin is a single location."""
    gravity = Gravity(deterrence_type="power_law", deterrence_params=[-1.0])

    # One origin, three destinations
    dist_matrix = np.array([[0.0, 1.0, 2.0]])  # shape (1, 3)
    origin_relevances = np.array([1.0])  # only one origin
    dest_relevances = np.array([10.0, 20.0, 30.0])  # three destinations

    location_idx = 0  # origin index (must be valid in dests as well)

    flows = gravity._compute_gravity_score(location_idx, dist_matrix, origin_relevances, dest_relevances)

    # Verify shape
    assert flows.shape == (1, 3)

    # Check self-flow (index 0) is zeroed out
    assert flows[0, location_idx] == 0.0

    # Other flows are greater than zero
    for j in range(flows.shape[1]):
        if j != location_idx:
            assert flows[0, j] > 0.0


def test_gravity_invalid_deterrence_type():
    """Initializing Gravity with an unsupported deterrence type should raise ValueError."""
    with pytest.raises(ValueError):
        Gravity(deterrence_type="invalid")
