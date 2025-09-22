import logging

import geopandas as gpd
import pandas as pd
from skmob.core.trajectorydataframe import TrajDataFrame
from skmob.models.sts_epr import STS_epr
from skmob.models.markov_diary_generator import MarkovDiaryGenerator

from geo_processor import GeoProcessor
from logger import Logger
from trajectory_processor import TrajectoryProcessor

from config_manager import ConfigManager

logging.getLogger().handlers.clear()

class STS_EPRTrajectory:
    """
    Class to simulate trajectories using the STS-EPR model from the scikit-mobility library. It initializes the model with
    the necessary parameters and provides a method to run the simulation. The generated trajectories are saved to files in the specified
    output directory. The class also logs key information about the simulation parameters.
    """
    def __init__(self, config_manager: ConfigManager, raw_trajectory: gpd.GeoDataFrame, tessellation: gpd.GeoDataFrame,
                 start_time: pd.Timestamp | None, end_time: pd.Timestamp | None, n_agents: int | None,
                 output_dir_path: str):
        """
        Initialize the STS_EPRTrajectory with necessary parameters.

        Args:
            config_manager (ConfigManager): Configuration manager for handling settings.
            raw_trajectory (gpd.GeoDataFrame): Raw trajectory data.
            tessellation (gpd.GeoDataFrame): Spatial tessellation for the area.
            start_time (pd.Timestamp | None): Start time for trajectory generation.
            end_time (pd.Timestamp | None): End time for trajectory generation.
            n_agents (int | None): Number of agents to simulate.
            output_dir_path (str): Directory path to save output files.
        """

        self.logger = Logger()
        self.config_manager = config_manager
        self.name = "STS_EPR"
        self.n_agents = n_agents
        self.start_time = start_time
        self.end_time = end_time
        self.output_dir_path = output_dir_path

        self.raw_trajectory = raw_trajectory
        self.tessellation = tessellation
        self.geo_processor = GeoProcessor()

    def process_generated_trajectory(self, synt_traj: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Process the generated trajectory to match the format of the original data. Assign tessellation IDs and resample
        to 1-hour intervals.

        Steps:

            1. Convert the generated trajectory to a GeoDataFrame and set the coordinate reference system (CRS).
            2. Perform a spatial join with the tessellation to assign tessellation IDs to each point.
            3. Rename columns to match the original data format.
            4. Resample the trajectory to 1-hour intervals, ensuring that each agent's trajectory is continuous.
            5. Create a new GeoDataFrame with the resampled data, including latitude and longitude columns.

        Args:
            synt_traj (gpd.GeoDataFrame): Generated trajectory from the STS-EPR model.
        Returns:
            gpd.GeoDataFrame: Processed trajectory with tessellation IDs and resampled to 1-hour intervals.
        """
        trajectory_processor = TrajectoryProcessor()

        generated_traj = synt_traj.to_geodataframe()
        generated_traj.set_crs(3857, allow_override=True, inplace=True)
        generated_traj_joined_with_tessellation = gpd.sjoin(generated_traj, self.tessellation, how="left",
                                                            predicate='intersects')
        generated_traj['tessellation_id'] = generated_traj_joined_with_tessellation['index_right']
        generated_traj.rename(columns={'uid': 'animal_id', 'datetime': 'time', 'lng': 'lon'}, inplace=True)
        generated_traj.set_index(['animal_id'], inplace=True)

        resampled_generated_traj = trajectory_processor.resample_time(generated_traj, '1h', 'tessellation_id')
        resampled_generated_traj.index.set_names(['animal_id', 'time'], inplace=True)
        gdf = gpd.GeoDataFrame(resampled_generated_traj,
                               geometry=gpd.points_from_xy(resampled_generated_traj.lon, resampled_generated_traj.lat),
                               crs=3857)
        gdf['lat'] = gdf.geometry.y
        gdf['lon'] = gdf.geometry.x
        return gdf

    def generate_trajectory(self) -> gpd.GeoDataFrame:
        """
        Generate synthetic trajectories using the STS-EPR model.

        Steps:

            1. Convert the raw trajectory data into a TrajDataFrame format required by the scikit-mobility library.
            2. Fit a Markov Diary Generator (MDG) to the trajectory data to model the temporal patterns of movement.
            3. Use the STS-EPR model to generate synthetic trajectories based on the fitted MDG and the provided tessellation.
            4. Process the generated trajectories to match the format of the original data, including assigning tessellation IDs and resampling to 1-hour intervals.
            5. Return the processed synthetic trajectories as a GeoDataFrame.

        Returns:
            gpd.GeoDataFrame: Processed synthetic trajectory generated by the STS-EPR model.
        """
        resampled_gdf_copy = self.raw_trajectory.reset_index().rename(columns={'level_0': 'animal_id', 'time': 'datetime'})
        trajectories_frame = TrajDataFrame(resampled_gdf_copy, latitude='lat', longitude='lon', user_id='animal_id', datetime='datetime')

        mdg = MarkovDiaryGenerator()
        mdg.fit(traj=trajectories_frame, n_individuals=self.n_agents, lid="labels")

        sts_epr = STS_epr()
        synt_traj = sts_epr.generate(start_date=self.start_time, end_date=self.end_time,
                                     spatial_tessellation=self.tessellation, diary_generator=mdg, n_agents=self.n_agents,
                                     relevance_column=self.config_manager.config.epr_params.relevance_column,
                                     show_progress=True)

        synt_traj = self.process_generated_trajectory(synt_traj)
        return synt_traj
