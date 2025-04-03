import matplotlib.pylab as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from compare_models.levy_flight.levy_flight import get_random_flight
from geo_processor import GeoProcessor
from distfit import distfit


class LevyFlightTrajectory:
    def __init__(self, name, filtered_data_means, tessellation, n_agents, start_time, steps, output_dir_path):
        self.name = name
        self.filtered_data_means = filtered_data_means
        self.tessellation = tessellation
        self.n_agents = n_agents
        self.start_time = start_time
        self.steps = steps
        self.output_dir_path = output_dir_path
        self.geo_processor = GeoProcessor()

    def compute_jump_length(self):
        jumps = self.geo_processor.jump_lengths(self.filtered_data_means)
        dfit = distfit(
            distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform',
                   'loggamma', 'truncexpon', 'truncnorm', 'truncpareto', 'powerlaw'], stats="wasserstein")

        # dfit = distfit(distr=['powerlaw'], stats="wasserstein")
        dfit.fit_transform(jumps[jumps != 0].dropna())

        dfit.plot()
        plt.loglog()
        plt.savefig(self.output_dir_path + '/levy_flight_jump_length_distribution.png')

        dfit.plot(
            pdf_properties={'color': '#472D30', 'linewidth': 4, 'linestyle': '--'},
            bar_properties=None,
            cii_properties=None,
            emp_properties={'color': '#E26D5C', 'linewidth': 0, 'marker': 'o'},
            figsize=(8, 5))
        plt.loglog()
        plt.savefig(self.output_dir_path + '/levy_flight_jump_length.png')
        return dfit

    def trim_trajectory(self, trajectory):
        trajectory.reset_index(inplace=True)
        if 'time' in trajectory.columns:
            trajectory = trajectory.sort_values(by=['animal_id', 'time'])
        first_nan_idx = trajectory.loc[trajectory['tessellation_id'].isna()].groupby('animal_id').head(1).index
        return trajectory.loc[trajectory.index < first_nan_idx.min()]

    def process_generated_trajectory(self, synt_traj):
        generated_traj = synt_traj
        generated_traj.set_crs(3857, allow_override=True, inplace=True)
        generated_traj_joined_with_tessellation = gpd.sjoin(generated_traj, self.tessellation, how="left",
                                                            predicate='intersects')
        generated_traj['tessellation_id'] = generated_traj_joined_with_tessellation['index_right']

        mask = generated_traj['tessellation_id'].notna()

        generated_traj.loc[mask, 'lon'] = self.tessellation.centroid.x.loc[
            generated_traj.loc[mask, 'tessellation_id']].values
        generated_traj.loc[mask, 'lat'] = self.tessellation.centroid.y.loc[
            generated_traj.loc[mask, 'tessellation_id']].values

        # generated_traj['lon'] = self.tessellation.centroid.x.loc[generated_traj['tessellation_id']].values
        # generated_traj['lat'] = self.tessellation.centroid.y.loc[generated_traj['tessellation_id']].values

        generated_traj.rename(columns={'uid': 'animal_id', 'datetime': 'time'}, inplace=True)
        gdf = gpd.GeoDataFrame(generated_traj,
                               geometry=gpd.points_from_xy(generated_traj.lon, generated_traj.lat),
                               crs=3857)
        return gdf

    def plot_trajectory(self, trajectory):
        trajectory = trajectory.reset_index()
        minx, miny, maxx, maxy = self.tessellation.total_bounds
        plt.figure(figsize=(8, 8))
        unique_animals = trajectory['animal_id'].unique()
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_animals)))
        for animal, color in zip(unique_animals, colors):
            animal_traj = trajectory[trajectory['animal_id'] == animal]
            plt.plot(animal_traj['lon'], animal_traj['lat'], 'o', markersize=3, alpha=0.5, label=f'Animal {animal}',
                     color=color)
        plt.xlabel(r'$X$')
        plt.ylabel(r'$Y$')
        x_margin = (maxx - minx) * 0.1
        y_margin = (maxy - miny) * 0.1
        plt.xlim(minx - x_margin, maxx + x_margin)
        plt.ylim(miny - y_margin, maxy + y_margin)
        plt.legend(title="Animal ID", loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(self.output_dir_path + '/levy_flight_trajectory.png')

    def generate_trajectory(self):
        model = self.compute_jump_length()
        minx, miny, maxx, maxy = self.tessellation.total_bounds

        dates = pd.date_range(self.start_time, periods=self.steps + 1, freq='H')

        trajectories = []

        for agent in range(self.n_agents):
            uid = agent
            x, y = get_random_flight(model, model.generate(self.steps), mode='2D', box_size_x=maxx - minx,
                                     box_size_y=maxy - miny,
                                     periodic=False)
            x = x + minx
            y = y + miny

            gdf = gpd.GeoDataFrame({'uid': uid, 'datetime': dates, 'geometry': gpd.points_from_xy(x, y)})
            gdf['lat'] = gdf.geometry.y
            gdf['lon'] = gdf.geometry.x
            gdf.set_crs(epsg=3857, inplace=True)

            synt_traj = self.process_generated_trajectory(gdf)
            trajectories.append(synt_traj)

        all_trajectories = pd.concat(trajectories, ignore_index=False)
        all_trajectories = all_trajectories[['animal_id', 'time', 'lat', 'lon', 'tessellation_id', 'geometry']]
        self.plot_trajectory(all_trajectories)
        return all_trajectories
