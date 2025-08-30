import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import distance
from shapely.geometry import Point


class GeoProcessor:
    """
    Class to handle geospatial processing of trajectory data. Includes methods for converting dataframes
    to geodataframes, computing grid sizes, assigning grid IDs, and calculating distances and jump lengths.
    """
    @staticmethod
    def convert_df_to_gdf(df: pd.DataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Convert a DataFrame with 'lat' and 'lon' columns to a GeoDataFrame with a specified CRS.
        Reprojects to EPSG:3857.

        Args:
            df (pd.DataFrame): DataFrame containing 'lat' and 'lon' columns.
            crs (str): Coordinate Reference System of the input DataFrame. Default is "EPSG:4326".
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with geometry column and reprojected to EPSG:3857.
        """
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=crs)
        gdf.to_crs(3857, inplace=True)
        gdf['lat'] = gdf.geometry.y
        gdf['lon'] = gdf.geometry.x
        return gdf

    @staticmethod
    def compute_grid_size(gdf: gpd.GeoDataFrame) -> int:
        """
        Compute an appropriate grid size based on the area covered by the GeoDataFrame. The grid size is determined
        by taking the square root of the area divided by 1000. This helps in creating a grid that is neither too fine
        nor too coarse for spatial analysis.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame for which to compute the grid size.
        Returns:
            int: Computed grid size.
        """
        area_size = (gdf.geometry.x.max() - gdf.geometry.x.min()) * (gdf.geometry.y.max() - gdf.geometry.y.min())
        return int(np.floor(np.sqrt(area_size / 1000)))

    @staticmethod
    def compute_points_in_grid(df: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compute the number of points from the GeoDataFrame that fall within each grid cell of the provided grid.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing points.
            grid (gpd.GeoDataFrame): GeoDataFrame representing the grid cells.
        Returns:
            gpd.GeoDataFrame: Grid GeoDataFrame with an additional 'points_count' column indicating the number of points
            in each cell.
        """
        joined = gpd.sjoin(df, grid, how='left', predicate='intersects')
        grid['points_count'] = grid.index.map(joined.groupby('tessellation_id').size()).fillna(0).astype(int)
        return grid

    def compute_mean_points_for_label(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Compute the mean latitude and longitude for each combination of 'animal_id' and 'labels' in the DataFrame.
        The resulting GeoDataFrame will have a geometry column representing these mean points.

        Args:
            df (pd.DataFrame): DataFrame containing 'animal_id', 'labels', 'lat', 'lon', and 'time' columns.
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with mean points for each 'animal_id' and 'labels' combination.
        """
        gdf = self.convert_df_to_gdf(df)
        gdf = gdf.set_index([gdf.index, 'time'])

        gdf['lat'] = gdf.groupby(['animal_id', 'labels'])['lat'].transform('mean')
        gdf['lon'] = gdf.groupby(['animal_id', 'labels'])['lon'].transform('mean')

        gdf['geometry'] = gdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        return gdf

    def trajectory_compression(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Compress the trajectory data by grouping consecutive points with the same geometry into single entries.
        This reduces the number of points in the trajectory while preserving the essential movement patterns.
        Each group will retain the first and last timestamps, as well as the associated labels.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing trajectory data with 'animal_id', 'time', 'labels', 'lat', 'lon', and 'geometry' columns.
        Returns:
            gpd.GeoDataFrame: Compressed GeoDataFrame with reduced trajectory points.
        """
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
        """
        Calculate the waiting time at each point in the trajectory as the difference between the end_time and the time.
        The waiting time is expressed in total seconds. The resulting DataFrame contains 'animal_id', 'time', and
        'waiting_time' columns. This information can be useful for understanding the duration of stops or pauses in
        movement.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing trajectory data with 'animal_id', 'time', and 'end_time' columns.
        Returns:
            pd.DataFrame: DataFrame with 'animal_id', 'time', and 'waiting_time' columns.
        """
        df["waiting_time"] = (df["end_time"] - df["time"]).dt.total_seconds()
        waiting_df = df[["animal_id", "time", "waiting_time"]]
        return waiting_df

    def get_starting_points(self, df: gpd.GeoDataFrame) -> list:
        """
        Get the starting grid IDs for each animal in the trajectory data. The starting point is defined as the first
        occurrence of each animal in the DataFrame. The function returns a list of starting grid IDs. This can be useful
        for initializing analyses or visualizations that require knowledge of the starting locations of the animals.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing trajectory data with 'animal_id' and 'tessellation_id' columns.
        Returns:
            list: List of starting grid IDs for each animal.
        """
        grouped = df.groupby(level=0).first()
        starting_grid_ids = grouped['tessellation_id'].tolist()
        return starting_grid_ids

    def assign_grid_id_to_points(self, df: gpd.GeoDataFrame, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assign grid IDs to points in the GeoDataFrame based on their spatial relationship with the provided grid.
        Uses a spatial join to match points with grid cells and assigns the corresponding 'tessellation_id' to each point.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing points to be assigned grid IDs.
            grid (gpd.GeoDataFrame): GeoDataFrame representing the grid cells with 'tessellation_id' column.
        Returns:
            gpd.GeoDataFrame: GeoDataFrame with an additional 'tessellation_id' column indicating the grid ID for each point.
        """
        joined = gpd.sjoin(df, grid, how='left', predicate='intersects')
        joined.drop(columns=['index_right'], inplace=True)
        if 'points_count' in joined.columns:
            joined.drop(columns=['points_count'], inplace=True)
        if 'tessellation_id' in joined.columns:
            joined['tessellation_id'] = joined['tessellation_id'].astype(int)
        return joined

    def compute_dist_matrix(self, df: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Compute a distance matrix between the centroids of the geometries in the GeoDataFrame. The resulting DataFrame
        contains the pairwise distances between each pair of centroids, with both rows and columns labeled by 'tessellation_id'.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing geometries with a 'tessellation_id' column.
        Returns:
            pd.DataFrame: DataFrame representing the distance matrix between centroids, indexed and columned by 'tessellation_id'.
        """
        if "tessellation_id" not in df.columns:
            raise ValueError("tessellation_id column is missing")
        df["centroid"] = df.geometry.centroid
        centroids = df["centroid"].apply(lambda point: (point.x, point.y)).tolist()
        dist_matrix = distance.cdist(centroids, centroids)
        return pd.DataFrame(dist_matrix, index=df["tessellation_id"], columns=df["tessellation_id"])

    def jump_lengths(self, df: gpd.GeoDataFrame) -> pd.Series:
        """
        Calculate the jump lengths between consecutive points in the trajectory for each animal. The jump length is
        defined as the distance between the current point and the previous point in the trajectory. The resulting Series
        contains the jump lengths, indexed by the original DataFrame's index.

        Args:
            df (gpd.GeoDataFrame): GeoDataFrame containing trajectory data with a geometry column.
        Returns:
            pd.Series: Series containing jump lengths between consecutive points, indexed by the original DataFrame's index.
        """
        jumps = df.dropna().geometry.groupby(level=0).progress_apply(lambda x: x.distance(x.shift()))
        return jumps
