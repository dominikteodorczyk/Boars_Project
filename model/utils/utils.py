import numpy as np
from pyproj import CRS
from math import sin, cos, atan2, sqrt, pi

EARTH_RADIUS = 6371.0


def get_geom_centroid(geom, return_lat_lng: bool = False) -> list[float]:
    x, y = geom.centroid.xy
    lng, lat = x[0], y[0]
    return [lat, lng] if return_lat_lng else [lng, lat]


def check_crs_is_metric(crs: CRS | str) -> bool:
    if isinstance(crs, str):
        crs = CRS.from_user_input(crs)
    return crs.is_projected and crs.axis_info[0].unit_name.lower() == 'metre'


def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def haversine_distance(loc1, loc2):
    lat1, lon1 = loc1
    lat2, lon2 = loc2

    lat1, lon1, lat2, lon2 = [x * pi / 180.0 for x in (lat1, lon1, lat2, lon2)]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS * c
