import geopandas as gpd
import pandas as pd
from compare_models.randomwalk.src.walkers import Walker

from logger import Logger


class RandomWalkTrajectory:
    def __init__(self, config_manager, grid_size, tessellation, n_agents, start_time, steps, output_dir_path):
        self.logger = Logger()
        self.config_manager = config_manager
        self.grid_size = grid_size
        self.tessellation = tessellation
        self.n_agents = n_agents
        self.start_time = start_time
        self.steps = steps
        self.output_path = output_dir_path

    def generate_trajectory(self):
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
