import copy
from typing import Optional
from numpy import ndarray
import rasterio as rs
import pandas as pd
import geopandas as gpd
from rasterio import CRS
import rasterio


class CoverageSampler:

    def __init__(self) -> None:
        self.raster_obj = rasterio.DatasetBase
        self.trajctories_data = pd.DataFrame()

    def read_raster(self, path:str, crs=None) -> None:
        self.raster_obj = rs.open(path)
        # self.raster_array = raster_obj.read()
        if crs:
            self.raster_crs = crs
        else:
            self.raster_crs = self.raster_obj.crs.to_epsg()

    def read_trajectiories(self, path:str, crs:int=4326, lon_col:str="lon", lat_col:str="lat"):
        self.trajctories_data = gpd.read_file(path)
        self.trajctories_data['geometry'] = gpd.points_from_xy(self.trajctories_data[lon_col], self.trajctories_data[lat_col])
        self.trajctories_data.set_crs(epsg=crs, inplace=True)
        self.trajctories_data.to_crs(epsg=self.raster_crs, inplace=True)

    def get_coverage(self) -> pd.DataFrame:
        coords = [(geom.x, geom.y) for geom in self.trajctories_data.geometry]
        values = [val[0] for val in self.raster_obj.sample(coords)]

        self.trajctories_data['coverage'] = values
        return self.trajctories_data





