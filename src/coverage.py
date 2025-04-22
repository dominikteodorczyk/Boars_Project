"""
Module for sampling raster coverage values along animal trajectories.

This module provides the CoverageSampler class, which allows you to:
- Load a raster file (e.g., land cover, NDVI, etc.)
- Load a set of GPS trajectories from file
- Sample the raster values at trajectory point locations
- Return a GeoDataFrame with coverage values for each point

Example:
    >>> from src.coverage import CoverageSampler
    >>> cs = CoverageSampler()
    >>> cs.read_raster("land_cover.tif", crs=2180)
    >>> cs.read_trajectiories("animals.csv", crs=4326)
    >>> results = cs.get_coverage()
"""

import pandas as pd
import geopandas as gpd
import rasterio as rs


class CoverageSampler:
    """
    A class for extracting raster coverage values for GPS trajectory points.

    This is useful for spatiotemporal ecological analyses, such as linking
    animal movement data with environmental variables (e.g., land cover,
    elevation).
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
