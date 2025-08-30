import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from distfit import distfit
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import linregress
from skmob.models.epr import DensityEPR

from geo_processor import GeoProcessor
from logger import Logger
from trajectory_processor import TrajectoryProcessor

logging.getLogger().handlers.clear()


class EPRTrajectory:
    def __init__(self, config_manager, filtered_data_means, raw_trajectory, tessellation, starting_positions,
                 start_time, end_time, n_agents, output_dir_path):
        self.logger = Logger()
        self.config_manager = config_manager
        self.name = "EPR"
        self.n_agents = n_agents
        self.start_time = start_time
        self.end_time = end_time
        self.output_dir_path = output_dir_path

        self.beta = None
        self.alpha = None
        self.gamma = None
        self.rho = None

        self.filtered_data_means = filtered_data_means
        self.raw_trajectory = raw_trajectory
        self.tessellation = tessellation
        self.starting_positions = starting_positions
        self.param_estimate()
        self.geo_processor = GeoProcessor()

    # Resample by 1-hour interval and select the label with the longest duration
    @staticmethod
    def longest_visited_row(groupa):
        """ Returns the row where the individual spent the longest time in an interval. """
        if groupa.empty:
            return pd.Series(dtype=object)  # Ensure an empty series is returned

        max_label = groupa.groupby('labels')['duration'].sum().idxmax()  # Find label with longest total duration

        # Select first row where this label appears
        row = groupa[groupa['labels'] == max_label].iloc[0]

        return row.T  # Return full row

    @staticmethod
    def truncated_power_law_pdf(x, beta, tau, xmin):
        x = np.array(x)
        return (beta / tau) * (x / xmin) ** (-beta - 1) * np.exp(-(x - xmin) / tau)

    def log_likelihood(self, params, data, xmin):
        beta, tau = params
        if beta <= 0 or tau <= 0:
            return np.inf

        ll = np.sum(np.log(self.truncated_power_law_pdf(data, beta, tau, xmin)))
        normalization, _ = quad(lambda x: self.truncated_power_law_pdf(x, beta, tau, xmin), xmin, np.inf)
        ll -= len(data) * np.log(normalization)

        return -ll

    def estimate_beta_tau_mle(self, data, xmin):
        initial_guess = [1.0, np.mean(data) - xmin]
        bounds = [(0.01, 10), (1, np.max(data))]

        result = minimize(self.log_likelihood, initial_guess, args=(data, xmin),
                          method='L-BFGS-B', bounds=bounds)

        if result.success:
            return result.x
        else:
            print(f"Optimization failed. Message: {result.message}")
            return None, None

    def param_estimate(self):
        to_concat = {}
        trajectory = self.raw_trajectory.copy()
        trajectory = trajectory.reset_index(level=1)
        for uid, group in trajectory.groupby(level=0):
            group = group[~group['time'].duplicated()]  # Remove duplicate timestamps within the same individual
            if len(group.labels.unique()) < 2:
                continue
            group.set_index('time', inplace=True)  # Set time as the index for time-based operations
            group['duration'] = (group.index.to_series().shift(-1) - group.index).dt.total_seconds()
            group['next'] = group.labels != group.labels.shift()
            group['moved'] = group.next.cumsum()
            (group.groupby('moved').duration.sum() / 3600).describe()
            group['duration'] = group['duration'].fillna(3600)

            group_resampled = group.resample('1h').apply(self.longest_visited_row).unstack()
            if group_resampled.index.nlevels > 1:
                group_resampled = group.resample('1h').apply(self.longest_visited_row)
            group_resampled = group_resampled.resample('1h').first()
            # Fill missing data using forward-fill and backward-fill
            group_resampled = group_resampled.drop(columns="geometry").ffill().bfill()

            # Store processed data for concatenation
            to_concat[uid] = group_resampled
        # Combine all processed individual datasets back into a single DataFrame
        df = pd.DataFrame(pd.concat(to_concat))
        if len(df.index.get_level_values(0).unique()) < 2:
            return
        # Identify when a new place is visited for each individual
        df['is_new'] = df.groupby(level=0, group_keys=False).apply(lambda x: ~x.labels.duplicated(keep='first'))

        # Compute cumulative number of distinct places visited (S(t)) for each individual
        df['new_sum'] = df.groupby(level=0).apply(lambda x: x.is_new.cumsum()).droplevel(1)

        # Compute the median number of observations per individual (to determine a common max time step) - cast to integer
        nrows = int(df.groupby(level=0).apply(lambda x: len(x)).min())

        # Prepare averaged data for population-level analysis
        # n_data: Sequence of step indices (1 to nrows)
        t_array = np.arange(1, nrows + 1)

        n_array = df.next.groupby(level=0).apply(lambda x: x.cumsum())
        n_array = np.array([n_array.groupby(level=0).nth(x).mean() for x in range(nrows)])

        # S_data: Mean S(n) across all individuals at each time step
        S_array = np.array([df.groupby(level=0)['new_sum'].nth(x).mean() for x in range(nrows)])

        # 1. Fit n(t) ~ t^beta - steps vs time
        log_t = np.log(t_array)
        log_n = np.log(n_array + 1e-10)
        slope_n, intercept_n, _, _, _ = linregress(log_t, log_n)
        beta_est = slope_n
        C_est = np.exp(intercept_n)  # n(t) ~ C_est * t^beta_est

        # 2. Fit S(t) ~ t^alpha
        log_S = np.log(S_array + 1e-10)
        slope_s, intercept_s, _, _, _ = linregress(log_t, log_S)
        alpha_est = slope_s
        K_est = np.exp(intercept_s)  # S(t) ~ K_est * t^alpha_est

        # 3. Compute gamma from alpha = beta / (1 + gamma)
        gamma_est = beta_est / alpha_est - 1

        # 4. Compute rho from the prefactor
        # We have: K_est = (1 + gamma) * [rho * C_est]^(1/(1 + gamma))
        # => ln(K_est) = ln(1 + gamma_est) + (1/(1 + gamma_est)) [ln(rho) + ln(C_est)]
        # Solve for rho:
        #    ln(rho) = (1 + gamma_est)* [ ln(K_est) - ln(1 + gamma_est) ] - ln(C_est)
        # (Some algebra rearranging the terms.)
        # It's often easier to do it step by step:

        num = np.log(K_est) - np.log(1 + gamma_est)
        den = 1.0 / (1.0 + gamma_est)
        # => num / den = ln(rho) + ln(C_est)
        temp = num / den - np.log(C_est)
        rho_est = np.exp(temp)

        # self.beta = beta_est
        # self.alpha = alpha_est
        self.gamma = gamma_est
        self.rho = rho_est

        self.logger.info("Estimated parameters:")
        self.logger.info(f"  beta (from n(t)):   {beta_est:.3f}")
        self.logger.info(f"  alpha (from S(t)):  {alpha_est:.3f}")
        self.logger.info(f"  gamma:              {gamma_est:.3f}")
        self.logger.info(f"  rho:                {rho_est:.3f}")

    def process_generated_trajectory(self, synt_traj):
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

    def find_best_fit(self):
        trajectory_compressed = self.geo_processor.trajectory_compression(self.filtered_data_means)
        waiting_times_df = self.geo_processor.waiting_times(trajectory_compressed)
        dfit = distfit(
            distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform',
                   'loggamma', 'truncexpon', 'truncnorm', 'truncpareto', 'powerlaw'], stats="wasserstein")
        dfit.fit_transform(waiting_times_df["waiting_time"])
        self.logger.info(f"Best fit: {dfit.model}")

        dfit.plot()
        plt.loglog()
        plt.savefig(f"{self.output_dir_path}/best_fit.png")

        dfit.plot(
            pdf_properties={'color': '#472D30', 'linewidth': 4, 'linestyle': '--'},
            bar_properties=None,
            cii_properties=None,
            emp_properties={'color': '#E26D5C', 'linewidth': 0, 'marker': 'o'},
            figsize=(8, 5))
        plt.loglog()
        plt.savefig(f"{self.output_dir_path}/best_fit_.png")
        return dfit

    def generate_trajectory(self):
        starting_points = self.starting_positions.copy()
        wt_model = self.find_best_fit()

        # estimated_beta, estimated_tau = self.estimate_beta_tau_mle(waiting_times.dropna().values ,min_wait_time_minutes)
        # self.logger.info("Estimated parameters KK:")
        # self.logger.info(f"Estimated beta: {self.beta}")
        # self.logger.info(f"Estimated tau: {self.tau}")

        epr = DensityEPR(wt_model=wt_model, rho=self.rho, gamma=self.gamma)
        synt_traj = epr.generate(start_date=self.start_time, end_date=self.end_time,
                                 spatial_tessellation=self.tessellation, n_agents=self.n_agents,
                                 starting_locations=starting_points,
                                 relevance_column=self.config_manager.config.epr_params.relevance_column,
                                 show_progress=True)
        synt_traj = self.process_generated_trajectory(synt_traj)
        return synt_traj
