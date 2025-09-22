import geopandas as gpd
import pandas as pd
from compare_models.randomwalk.src.walkers import Walker

from logger import Logger
from config_manager import ConfigManager


class RandomWalkTrajectory:
    """
    Class to generate synthetic trajectories based on a random walk model. It simulates the movement of a specified
    number of agents over a defined time period, using a given grid size and tessellation for spatial referencing.
    """

    def __init__(self, config_manager: ConfigManager, grid_size: int, tessellation: gpd.GeoDataFrame, n_agents: int,
                 start_time: pd.Timestamp, steps: int, output_dir_path: str) -> None:
        """
        Initialize the RandomWalkTrajectory with necessary parameters.

        Args:
            config_manager (ConfigManager): Configuration manager for handling settings.
            grid_size (int): Size of the grid for the random walk.
            tessellation (gpd.GeoDataFrame): Geospatial tessellation for spatial referencing.
            n_agents (int): Number of agents to simulate.
            start_time (pd.Timestamp): Start time for the trajectory simulation.
            steps (int): Number of time steps for the simulation.
            output_dir_path (str): Directory path to save output files and plots.
        """
        self.logger = Logger()
        self.config_manager = config_manager
        self.grid_size = grid_size
        self.tessellation = tessellation
        self.n_agents = n_agents
        self.start_time = start_time
        self.steps = steps
        self.output_path = output_dir_path

    def generate_trajectory(self) -> gpd.GeoDataFrame:
        """
        Generate synthetic trajectories for the specified number of agents using a random walk model.
        Each agent starts at a random position and moves according to the defined movement pattern and step size.
        The generated trajectories are compiled into a GeoDataFrame with appropriate spatial referencing.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame containing the generated trajectories with geometry column.
        """
        move_pattern = "true"
        random_start = "true"

        concated_trajectory = []
        for walker_num in range(self.n_agents):
            new_walker = Walker(agent_id=walker_num, start_time=self.start_time.strftime('%Y-%m-%d %H:%M:%S'), total_steps=self.steps, step_size=self.grid_size, tessellation=self.tessellation)
            new_walker.random_walk(move_pattern, random_start)
            trajectory_df = pd.DataFrame(new_walker.trajectory, columns=["animal_id", "time", "lat", "lon", "tessellation_id"])
            concated_trajectory.append(trajectory_df)

        concated_trajectory = pd.concat(concated_trajectory)
        gdf = gpd.GeoDataFrame(concated_trajectory, geometry=gpd.points_from_xy(concated_trajectory.lat, concated_trajectory.lon))
        gdf.crs = "EPSG:3857"
        return gdf
