import numpy as np
from pyproj import CRS
from math import sin, cos, atan2, sqrt, pi

EARTH_RADIUS = 6371.0


def get_geom_centroid(geom, return_lat_lng: bool = False) -> list[float]:
    """
    Get the centroid of a geometry as [lng, lat] or [lat, lng]. Defaults to [lng, lat].

    Args:
        geom: A shapely geometry object.
        return_lat_lng (bool): If True, returns [lat, lng]. If False, returns [lng, lat].
    Returns:
        list[float]: Centroid coordinates.
    """
    x, y = geom.centroid.xy
    lng, lat = x[0], y[0]
    return [lat, lng] if return_lat_lng else [lng, lat]


def check_crs_is_metric(crs: CRS | str) -> bool:
    """
    Check if a given CRS is projected and uses metric units (meters).

    Args:
        crs (CRS | str): A pyproj CRS object or a string representation of a CRS.
    Returns:
        bool: True if the CRS is projected and uses meters, False otherwise.
    """
    if isinstance(crs, str):
        crs = CRS.from_user_input(crs)
    return crs.is_projected and crs.axis_info[0].unit_name.lower() == 'metre'


def euclidean_distance(point1: list, point2: list) -> float:
    """
    Compute the Euclidean distance between two points in 2D space.

    Args:
        point1 (list): Coordinates of the first point [x1, y1].
        point2 (list): Coordinates of the second point [x2, y2].
    Returns:
        float: Euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def haversine_distance(loc1: list, loc2: list) -> float:
    """
    Calculate the Haversine distance between two points on the Earth's surface specified in decimal degrees.

    Args:
        loc1 (list): [latitude, longitude] of the first location in decimal degrees.
        loc2 (list): [latitude, longitude] of the second location in decimal degrees.
    Returns:
        float: Distance between the two locations in kilometers.
    """
    lat1, lon1 = loc1
    lat2, lon2 = loc2

    lat1, lon1, lat2, lon2 = [x * pi / 180.0 for x in (lat1, lon1, lat2, lon2)]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS * c
