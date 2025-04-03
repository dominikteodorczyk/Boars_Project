import os

from humobi.misc.create_grid import create_grid

from config_manager import ConfigManager
from data_loader import DataLoader
from file_manager import FileManager
from geo_processor import GeoProcessor
from logger import Logger
from trajectory_processor import TrajectoryProcessor
from trajectory_simulator import TrajectorySimulator
from utils import compute_emd, compute_flows, plot_emd, plot_flows


class Comparator:
    def __init__(self, config_file: str):
        self.logger = Logger()
        self.logger.info("Initializing Comparator")

        self.config_manager = ConfigManager(config_file)
        self.file_manager = FileManager(self.config_manager.config.paths.output_dir, self.logger)
        self.data_loader = DataLoader()
        self.geo_processor = GeoProcessor()
        self.trajectory_processor = TrajectoryProcessor()
        self.trajectory_simulator = None

    def run(self):
        for root, _, files in os.walk(self.config_manager.config.paths.input_dir):
            #TODO: Remove splicing limit
            # for file in files[0:1]:
            for file in files:
                self.logger.info(f"Processing file: {file}")
                file_path = os.path.join(root, file)
                file_name = os.path.splitext(file)[0]
                output_dir = self.file_manager.create_output_dir_for_file(file_name)

                data = self.data_loader.read_raw_data(str(file_path))
                self.logger.info(f"Number of unique animals: {len(data.index.unique())}")
                self.logger.info(f"Number of rows in input data: {len(data)}")
                filtered_data = self.trajectory_processor.filter_by_quartile(data,
                                                                             self.config_manager.config.input_file_params.quartile)

                self.logger.info(f"Number of unique animals after filtering: {len(filtered_data.index.unique())}")
                self.logger.info(f"Number of rows in filtered data: {len(filtered_data)}")

                filtered_data_means = self.geo_processor.compute_mean_points_for_label(filtered_data)

                resampled_data = self.trajectory_processor.resample_time(filtered_data,
                                                                         self.config_manager.config.input_file_params.resample_freq)
                resampled_gdf = self.geo_processor.convert_df_to_gdf(resampled_data)
                resampled_gdf.to_file(os.path.join(output_dir, f'resampled_{file_name}.geojson'), driver='GeoJSON')

                grid_size = self.geo_processor.compute_grid_size(resampled_gdf)
                self.logger.info(f"Grid size: {grid_size}")

                tessellation = create_grid(resampled_gdf, grid_size)
                tessellation["tessellation_id"] = tessellation.index.astype(int)
                raw_tessellation = tessellation.copy()
                tessellation = self.geo_processor.compute_points_in_grid(resampled_gdf, tessellation)
                self.logger.info(f"Number of grids: {len(tessellation)}")
                tessellation.to_file(os.path.join(output_dir, 'tessellation.geojson'), driver='GeoJSON')
                self.logger.info(f"Saved tessellation to file: {os.path.join(output_dir, 'tessellation.geojson')}")

                trajectory_with_grid_id = self.geo_processor.assign_grid_id_to_points(resampled_gdf, tessellation)
                starting_positions = self.geo_processor.get_starting_points(trajectory_with_grid_id)[::-1]

                self.trajectory_simulator = TrajectorySimulator(filtered_data_means, resampled_gdf, grid_size,
                                                                tessellation, starting_positions, output_dir, file_name,
                                                                self.config_manager)

                epr_traj = self.trajectory_simulator.simulate("EPR")
                rw_traj = self.trajectory_simulator.simulate("RandomWalk")
                lf_traj = self.trajectory_simulator.simulate("LevyFlight")

                flows_org = compute_flows(trajectory_with_grid_id, 'all')
                flows_epr = compute_flows(epr_traj, 'all')
                flows_rw = compute_flows(rw_traj, 'all')
                flows_lf = compute_flows(lf_traj, 'all')

                plot_flows(flows_org, flows_epr, 25, output_dir, "epr_flows", 'logscale')
                plot_flows(flows_org, flows_rw, 25, output_dir, "rw_flows", 'logscale')
                plot_flows(flows_org, flows_lf, 25, output_dir, "lf_flows", 'logscale')

                emd_epr = compute_emd(trajectory_with_grid_id, epr_traj, raw_tessellation)
                emd_rw = compute_emd(trajectory_with_grid_id, rw_traj, raw_tessellation)
                emd_lf = compute_emd(trajectory_with_grid_id, lf_traj, raw_tessellation)

                plot_emd([emd_epr, emd_rw, emd_lf], ["EPR", "RandomWalk", "LevyFlight"], output_dir, "emd_rw_lf")


if __name__ == "__main__":
    comparator = Comparator(config_file="config.yaml")
    comparator.run()
