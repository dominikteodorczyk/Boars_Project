"""
Module for sampling raster coverage values along animal trajectories.

This module provides the CoverageSampler class, which allows you to:
- Load a raster file (e.g., land cover, NDVI, etc.)
- Load a set of GPS trajectories from file
- Sample the raster values at trajectory point locations
- Return a GeoDataFrame with coverage values for each point
"""

import pandas as pd
import geopandas as gpd
import rasterio as rs
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from geopy.distance import geodesic
from tqdm import tqdm


class CoverageSampler:
    """
    A class for extracting raster coverage values for GPS trajectory points.

    This is useful for spatiotemporal ecological analyses, such as linking
    animal movement data with environmental variables (e.g., land cover,
    elevation).

    Example:
    >>> cs = CoverageSampler()
    >>> cs.read_raster("land_cover.tif", crs=2180)
    >>> cs.read_trajectiories("animals.csv", crs=4326)
    >>> results = cs.get_coverage()
    """
    def __init__(self) -> None:
        self.raster_obj = rs.DatasetBase
        self.trajctories_data = pd.DataFrame()

    def read_raster(self, path: str, crs=None) -> None:
        """
        Loads a raster file and sets the coordinate reference system (CRS).

        Args:
            path (str): Path to the raster file (e.g., GeoTIFF).
            crs (int, optional): EPSG code to override the raster's CRS.
            If not provided, the CRS is taken from the file itself.
        """
        self.raster_obj = rs.open(path)
        # self.raster_array = raster_obj.read()
        if crs:
            self.raster_crs = crs
        else:
            self.raster_crs = self.raster_obj.crs.to_epsg()

    def read_trajectiories(
        self,
        path: str,
        crs: int = 4326,
        lon_col: str = "lon",
        lat_col: str = "lat"
    ):
        """
        Loads trajectory data and transforms it into the raster's CRS.

        Args:
            path (str): Path to a file containing trajectory data (e.g., CSV,
                GeoJSON, shapefile).
            crs (int): EPSG code for the original CRS
                of the input file (default is 4326 - WGS84).
            lon_col (str): Column name for longitude values.
            lat_col (str): Column name for latitude values.
        """
        self.trajctories_data = gpd.read_file(path)
        self.trajctories_data["geometry"] = gpd.points_from_xy(
            self.trajctories_data[lon_col], self.trajctories_data[lat_col]
        )
        self.trajctories_data.set_crs(epsg=crs, inplace=True)
        self.trajctories_data.to_crs(epsg=self.raster_crs, inplace=True)

    def get_coverage(self) -> pd.DataFrame:
        """
        Samples raster values at each trajectory point.

        Returns:
            pd.DataFrame: The original trajectory data with
                an additional 'coverage' column containing sampled
                raster values.
        """
        coords = [(geom.x, geom.y) for geom in self.trajctories_data.geometry]  # type: ignore
        values = [val[0] for val in self.raster_obj.sample(coords)]

        self.trajctories_data["coverage"] = values
        return self.trajctories_data

class WeatherCoverage:
    """
    A class for interpolating weather parameters (e.g., temperature, humidity,
    wind, rainfall) along animal trajectories using weather station data.

    The interpolation method used is IDW (Inverse Distance Weighting).
    """
    def __init__(self, data_path, weather_path) -> None:
        """
        Initialize the WeatherCoverage object with animal trajectory
        and weather station data.

        Args:
            data_path (str): Path to a CSV file containing trajectory data
                (must include at least user_id, datetime, lon, lat).
            weather_path (str): Path to a CSV file containing weather station data
                (must include location_id, latitude, longitude, time,
                and weather parameters).
        """
        self.weather_data = pd.read_csv(weather_path)
        self.trajctories_data = pd.read_csv(data_path)

        self.trajctories_data["datetime"] = pd.to_datetime(self.trajctories_data["datetime"])

        self.weather_data["datetime"] = pd.to_datetime(self.weather_data['time'])
        self.weather_data.drop("time", axis=1, inplace=True)

        self.locations = self.weather_data[['location_id','latitude','longitude']].drop_duplicates()
        self.locations = self.locations.set_index('location_id').reset_index()

        self.animal_data_rounded = self.trajctories_data
        self.animal_data_rounded['rounded_down'] = self.animal_data_rounded['datetime'].dt.floor('H')
        self.animal_data_rounded['rounded_up'] = self.animal_data_rounded['datetime'].dt.ceil('H')
        self.params = [x for x in self.weather_data.columns if x not in ['Unnamed: 0','location_id','latitude','longitude','datetime']]

    def idw_interpolation(self, lat, lon, stations_lat, stations_lon, stations_param, power=2):
        """
        Perform Inverse Distance Weighting (IDW) interpolation for a given parameter.

        Args:
            lat (float): Latitude of the target point.
            lon (float): Longitude of the target point.
            stations_lat (array-like): Latitudes of weather stations.
            stations_lon (array-like): Longitudes of weather stations.
            stations_param (array-like): Values of the weather parameter at stations.
            power (int, optional): Power parameter for IDW (default is 2).

        Returns:
            float: Interpolated parameter value at the target location.
        """
        numerator = 0
        denominator = 0
        for lati, loni, parami in zip(stations_lat, stations_lon, stations_param):
            distance = geodesic(
                    (lat, lati),
                    (lon, loni)
                ).meters
            if distance == 0:
                return parami
            weight = 1 / (distance ** power)
            numerator += weight * parami
            denominator += weight
        return numerator / denominator

    def process_row(self, row):
        """
        Process a single trajectory row by interpolating weather parameters
        for the given location and time.

        Args:
            row (pandas.Series or namedtuple): A single row from the trajectory
                DataFrame containing at least datetime, rounded_down, rounded_up,
                lon, lat, and user_id.

        Returns:
            dict: A dictionary containing user_id, datetime, lon, lat,
            and interpolated weather parameters.
        """
        df_below = self.weather_data[self.weather_data["datetime"] == row.rounded_down].copy()
        df_above = self.weather_data[self.weather_data["datetime"] == row.rounded_up].copy()

        df_below.drop(['latitude', 'longitude', 'datetime', 'Unnamed: 0'], axis=1, inplace=True)
        df_above.drop(['latitude', 'longitude', 'datetime', 'Unnamed: 0'], axis=1, inplace=True)
        df_below = df_below.set_index("location_id")
        df_above = df_above.set_index("location_id")

        time_ratio = 0 if row.rounded_down == row.rounded_up else \
            (row.datetime - row.rounded_down).total_seconds() / (row.rounded_up - row.rounded_down).total_seconds()

        df_diff = df_below.reset_index(drop=True) + ((df_above.reset_index(drop=True) - df_below.reset_index(drop=True)) * time_ratio)
        df_diff = df_diff.join(self.locations[['latitude', 'longitude']])

        fix_dict = {"user_id": row.user_id, "datetime": row.datetime, "lon": row.lon, "lat": row.lat}

        for param in self.params:
            lats = df_diff['latitude'].values
            lons = df_diff['longitude'].values
            ok_param = df_diff[param].values
            if np.all(ok_param == 0) or np.isclose(np.std(ok_param), 0):
                fix_dict[param] = ok_param[0]
            else:
                try:
                    interpolated_param = self.idw_interpolation(row.lat, row.lon, lats, lons, ok_param, power=2)
                    fix_dict[param] = interpolated_param
                except Exception as e:
                    print("Interpolacja nie powiodła się:", param, e)
                    fix_dict[param] = np.nan
        return fix_dict

    def interpolate_by_idw(self):
        """
        Run IDW interpolation for all trajectory rows using multithreading.

        Uses ThreadPoolExecutor to parallelize the processing of rows.

        Returns:
            pandas.DataFrame: A DataFrame containing trajectory information
            along with interpolated weather parameters for each row.
        """
        results = []
        with ThreadPoolExecutor(max_workers=28) as executor:
            futures = [executor.submit(self.process_row, row) for row in self.animal_data_rounded.itertuples(index=False)]
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())


        return pd.DataFrame(results)
