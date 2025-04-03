import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import distance
from shapely.geometry import Point


class GeoProcessor:
    @staticmethod
    def convert_df_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
        gdf.to_crs(3857, inplace=True)
        gdf['lat'] = gdf.geometry.y
        gdf['lon'] = gdf.geometry.x
        return gdf

    @staticmethod
    def compute_grid_size(gdf: gpd.GeoDataFrame):
        area_size = (gdf.geometry.x.max() - gdf.geometry.x.min()) * (gdf.geometry.y.max() - gdf.geometry.y.min())
        return int(np.floor(np.sqrt(area_size / 1000)))

    @staticmethod
    def compute_points_in_grid(df: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        joined = gpd.sjoin(df, grid, how='left', predicate='intersects')
        grid['points_count'] = grid.index.map(joined.groupby('tessellation_id').size()).fillna(0).astype(int)
        return grid

    def compute_mean_points_for_label(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        gdf = self.convert_df_to_gdf(df)
        gdf = gdf.set_index([gdf.index, 'time'])

        gdf['lat'] = gdf.groupby(['animal_id', 'labels'])['lat'].transform('mean')
        gdf['lon'] = gdf.groupby(['animal_id', 'labels'])['lon'].transform('mean')

        gdf['geometry'] = gdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        return gdf

    def trajectory_compression(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        df = df.reset_index()
        df["location_group"] = (df["geometry"] != df["geometry"].shift()).cumsum()

        return df.groupby(["animal_id", "location_group"]).agg(
            animal_id=("animal_id", "first"),
            time=("time", "first"),
            end_time=("time", "last"),
            labels=("labels", "first"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            geometry=("geometry", "first")
        ).reset_index(drop=True)

    def waiting_times(self, df: gpd.GeoDataFrame) -> pd.DataFrame:
        df["waiting_time"] = (df["end_time"] - df["time"]).dt.total_seconds()
        waiting_df = df[["animal_id", "time", "waiting_time"]]
        return waiting_df

    def get_starting_points(self, df: gpd.GeoDataFrame) -> list:
        grouped = df.groupby(level=0).first()
        starting_grid_ids = grouped['tessellation_id'].tolist()
        return starting_grid_ids

    def assign_grid_id_to_points(self, df: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        joined = gpd.sjoin(df, grid, how='left', predicate='intersects')
        joined.drop(columns=['index_right'], inplace=True)
        if 'points_count' in joined.columns:
            joined.drop(columns=['points_count'], inplace=True)
        if 'tessellation_id' in joined.columns:
            joined['tessellation_id'] = joined['tessellation_id'].astype(int)
        return joined

    def compute_dist_matrix(self, df: gpd.GeoDataFrame) -> pd.DataFrame:
        if "tessellation_id" not in df.columns:
            raise ValueError("tessellation_id column is missing")
        df["centroid"] = df.geometry.centroid
        centroids = df["centroid"].apply(lambda point: (point.x, point.y)).tolist()
        dist_matrix = distance.cdist(centroids, centroids)
        return pd.DataFrame(dist_matrix, index=df["tessellation_id"], columns=df["tessellation_id"])

    def jump_lengths(self, df: gpd.GeoDataFrame) -> pd.Series:
        jumps = df.dropna().geometry.groupby(level=0).progress_apply(lambda x: x.distance(x.shift()))
        return jumps
