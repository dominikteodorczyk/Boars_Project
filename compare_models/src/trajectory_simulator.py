import os
import geopandas as gpd
import pandas as pd
from humobi.tools.user_statistics import user_trajectories_duration

from config_manager import ConfigManager
from epr_trajectory import EPRTrajectory
from sts_epr_trajecotry import STS_EPRTrajectory
from logger import Logger
from random_walk_trajectory import RandomWalkTrajectory
from levy_flight_trajectory import LevyFlightTrajectory


class TrajectorySimulator:
    """
    Class to simulate trajectories using different models (EPR, Random Walk, Levy Flight). It initializes the models with
    the necessary parameters and provides a method to run the simulation for a specified model. The generated trajectories
    are saved to files in the specified output directory. The class also logs key information about the simulation parameters.
    Methods:
    - __init__: Initializes the simulator with filtered data, resampled data, grid size, tessellation, starting positions, output directory, file name, and configuration manager.
    - get_users_list: Extracts a list of unique user IDs from the DataFrame.
    - get_min_max_date: Retrieves the minimum and maximum timestamps from the DataFrame.
    - get_params: Computes and sets the number of agents, start time, end time, and number of steps for the simulation based on the resampled data.
    - simulate: Runs the simulation for the specified model (EPR, Random Walk, Levy Flight) and saves the generated trajectory to a file.
    """

    def __init__(self, filtered_data_means: gpd.GeoDataFrame, resampled_gdf: gpd.GeoDataFrame, grid_size: int,
                 tessellation: gpd.GeoDataFrame, starting_positions: list, output_dir_path: str, file_name: str,
                 config_manager: ConfigManager) -> None:
        """
        Initialize the TrajectorySimulator with necessary parameters.

        Args:
            filtered_data_means (gpd.GeoDataFrame): Filtered trajectory data with mean points.
            resampled_gdf (gpd.GeoDataFrame): Resampled trajectory data.
            grid_size (int): Size of the grid for spatial referencing.
            tessellation (gpd.GeoDataFrame): Geospatial tessellation for spatial referencing.
            starting_positions (list): List of starting grid IDs for each animal.
            output_dir_path (str): Directory path to save output files and plots.
            file_name (str): Base name for output files.
            config_manager (ConfigManager): Configuration manager for handling settings.
        """
        self.logger = Logger()
        self.filtered_data_means = filtered_data_means
        self.resampled_gdf = resampled_gdf
        self.grid_size = grid_size
        self.output_dir_path = output_dir_path
        self.file_name = file_name

        self.n_agents = None
        self.start_time = None
        self.end_time = None
        self.steps = None

        self.get_params()

        self.logger.info(f"Number of animals to generate: {self.n_agents}")
        self.logger.info(f"Start time: {self.start_time}")
        self.logger.info(f"End time: {self.end_time}")
        self.logger.info(f"Number of steps: {self.steps}")

        self.epr_trajectory = EPRTrajectory(config_manager, filtered_data_means, resampled_gdf, tessellation, starting_positions, self.start_time, self.end_time, self.n_agents, self.output_dir_path)
        self.random_walk_trajectory = RandomWalkTrajectory(config_manager, grid_size, tessellation, self.n_agents, self.start_time, self.steps, self.output_dir_path)
        self.levy_flight_trajectory = LevyFlightTrajectory("LevyFlight", filtered_data_means, tessellation, self.n_agents, self.start_time, self.steps, self.output_dir_path)
        self.sts_epr_trajectory = STS_EPRTrajectory(config_manager, resampled_gdf, tessellation, self.start_time, self.end_time, self.n_agents, self.output_dir_path)

    def get_users_list(self, df: pd.DataFrame) -> list:
        """
        Extract a list of unique user IDs from the DataFrame. The user IDs are assumed to be in the first level of the DataFrame's index.

        Args:
            df (pd.DataFrame): DataFrame containing trajectory data with a multi-level index where the first level represents user IDs.
        Returns:
            list: List of unique user IDs.
        """
        return list(df.groupby(level=0).groups.keys())

    def get_min_max_date(self, df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the minimum and maximum timestamps from the DataFrame. The timestamps are assumed to be in the 'time' level of the DataFrame's index.

        Args:
            df (pd.DataFrame): DataFrame containing trajectory data with a multi-level index where one level is 'time'.
        Returns:
            tuple[pd.Timestamp, pd.Timestamp]: A tuple containing the minimum and maximum timestamps.
        """
        min_date = df.index.get_level_values('time').min()
        max_date = df.index.get_level_values('time').max()
        return min_date, max_date

    def get_params(self) -> None:
        """
        Compute and set the number of agents, start time, end time, and number of steps for the simulation based on the resampled data.
        The number of agents is determined by the number of unique users in the resampled data. The start time is set to the minimum timestamp,
        and the end time is calculated by adding the total duration of the longest user trajectory (in hours) to the start time. The number of steps is set to the total duration in hours.

        Args:
            None
        Returns:
            None
        """
        number_of_hours = max(user_trajectories_duration(self.resampled_gdf, 'h'))
        self.n_agents = len(self.get_users_list(self.resampled_gdf))
        min_date, max_date = self.get_min_max_date(self.resampled_gdf)

        self.start_time = min_date
        self.end_time = self.start_time + pd.Timedelta('1h') * number_of_hours
        self.steps = number_of_hours

    def simulate(self, model_name: str) -> gpd.GeoDataFrame:
        """
        Run the simulation for the specified model (EPR, STS-EPR, Random Walk, Levy Flight) and save the generated trajectory to a file. The generated trajectory is returned as a GeoDataFrame.
        When saving the trajectory to a file, the filename is prefixed with the model name (e.g., 'epr_generated_', 'sts_epr_generated_', 'rw_generated_', 'lf_generated_') followed by the original file name and a '.geojson' extension.

        Args:
            model_name (str): Name of the trajectory model to simulate ('EPR', 'STS-EPR', 'RandomWalk', 'LevyFlight').
        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the generated trajectory with geometry column.
        """
        if model_name == "EPR":
            trajectory = self.epr_trajectory.generate_trajectory()
            trajectory.to_file(os.path.join(self.output_dir_path, 'epr_generated_' + self.file_name + '.geojson'),
                        driver='GeoJSON')
            return trajectory
        elif model_name == "RandomWalk":
            trajectory = self.random_walk_trajectory.generate_trajectory()
            trajectory.set_index(['animal_id', 'time'], inplace=True)
            trajectory.to_file(os.path.join(self.output_dir_path, 'rw_generated_' + self.file_name + '.geojson'), driver='GeoJSON')
            return trajectory
        elif model_name == "LevyFlight":
            trajectory = self.levy_flight_trajectory.generate_trajectory()
            trajectory.set_index(['animal_id', 'time'], inplace=True)
            trajectory.to_file(os.path.join(self.output_dir_path, 'lf_generated_' + self.file_name + '.geojson'), driver='GeoJSON')
            return trajectory
        elif model_name == "STS_EPR":
            trajectory = self.sts_epr_trajectory.generate_trajectory()
            trajectory.to_file(os.path.join(self.output_dir_path, 'sts_epr_generated_' + self.file_name + '.geojson'), driver='GeoJSON')
            return trajectory
