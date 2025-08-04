import numpy as np
from shapely.geometry import Point
from pyproj import CRS
from model.utils.utils import get_geom_centroid, check_crs_is_metric, euclidean_distance, haversine_distance


def test_returns_centroid_in_lat_lng_order_when_flag_true():
    geom = Point(10, 20)
    result = get_geom_centroid(geom, return_lat_lng=True)
    assert np.allclose(result, [20, 10])


def test_returns_centroid_in_lng_lat_order_when_flag_false():
    geom = Point(10, 20)
    result = get_geom_centroid(geom, return_lat_lng=False)
    assert np.allclose(result, [10, 20])


def test_returns_true_for_metric_crs():
    crs = CRS.from_epsg(3857)
    assert check_crs_is_metric(crs) is True


def test_returns_false_for_geographic_crs():
    crs = CRS.from_epsg(4326)
    assert check_crs_is_metric(crs) is False


def test_accepts_crs_as_string_and_returns_true_for_metric():
    assert check_crs_is_metric("EPSG:3857") is True


def test_calculates_euclidean_distance_between_two_points():
    p1 = (0, 0)
    p2 = (3, 4)
    assert euclidean_distance(p1, p2) == 5.0


def test_returns_zero_for_euclidean_distance_of_same_point():
    p = (1.5, -2.5)
    assert euclidean_distance(p, p) == 0.0


def test_calculates_haversine_distance_between_two_known_points():
    loc1 = (0, 0)
    loc2 = (0, 1)
    result = haversine_distance(loc1, loc2)
    assert np.isclose(result, 111.19, atol=0.1)


def test_returns_zero_for_haversine_distance_of_same_point():
    loc = (52.2297, 21.0122)
    assert haversine_distance(loc, loc) == 0.0


def test_handles_negative_coordinates_in_haversine_distance():
    loc1 = (-45, -45)
    loc2 = (45, 45)
    result = haversine_distance(loc1, loc2)
    assert result > 0
