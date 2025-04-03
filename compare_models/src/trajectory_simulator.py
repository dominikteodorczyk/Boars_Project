import os

import pandas as pd
from humobi.tools.user_statistics import user_trajectories_duration

from epr_trajectory import EPRTrajectory
from logger import Logger
from random_walk_trajectory import RandomWalkTrajectory
from levy_flight_trajectory import LevyFlightTrajectory


class TrajectorySimulator:
    def __init__(self, filtered_data_means, resampled_gdf, grid_size, tessellation, starting_positions, output_dir_path, file_name, config_manager):
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

    def get_users_list(self, df: pd.DataFrame):
        return list(df.groupby(level=0).groups.keys())

    def get_min_max_date(self, df: pd.DataFrame):
        min_date = df.index.get_level_values('time').min()
        max_date = df.index.get_level_values('time').max()
        return min_date, max_date

    def get_params(self):
        number_of_hours = max(user_trajectories_duration(self.resampled_gdf, 'h'))
        self.n_agents = len(self.get_users_list(self.resampled_gdf))
        min_date, max_date = self.get_min_max_date(self.resampled_gdf)

        self.start_time = min_date
        self.end_time = self.start_time + pd.Timedelta('1h') * number_of_hours
        self.steps = number_of_hours

    def simulate(self, model_name):
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
