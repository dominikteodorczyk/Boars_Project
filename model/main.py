import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Pandas doesn't allow columns to be created via a new attribute name")

from humobi.structures.trajectory import TrajectoriesFrame
from src.markov_chain import MarkovChain, MarkovChainConfig
from src.EPR import EPR, EPRConfig, Ditras

from pydantic import ValidationError
import pandas as pd
import geopandas as gpd
from box import Box
import yaml
import os
import shutil
from datetime import datetime

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config_box = Box.from_yaml(filename=config_path, Loader=yaml.FullLoader)

file_path = config_box.general.file_path
trajectories_frame = TrajectoriesFrame(file_path)
output_path = config_box.general.output_path

tessellation_path = config_box.general.tessellation_path
tessellation = gpd.GeoDataFrame.from_file(tessellation_path)

start_time = pd.to_datetime(config_box.general.start_simulation)
end_time = pd.to_datetime(config_box.general.end_simulation)
random_seed = config_box.general.random_seed

run_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')

if config_box.execution.run_markov:
    print("🚀 Initialization and fitting of the Markov Chain model...")
    try:
        markov_config = MarkovChainConfig(**config_box.markov)
        mc = MarkovChain(config=markov_config)
        mc.fit(trajectories_frame)
        print("✅ Markov Chain model completed successfully.")

        if config_box.markov.generate.enabled:
            print("🔄 Generating Markov Chain diary...")
            diary = mc.generate(
                duration_hours=config_box.markov.generate.duration_hours,
                start_date=start_time,
                seed=random_seed
            )
            print(f"✅ Diary generated successfully.")
    except ValidationError as e:
        print(f"❌ Error in Markov Chain configuration:\n{e}")
        exit(1)

if config_box.execution.run_epr:
    print("🚀 Initialization and generation trajectory using the EPR model...")
    try:
        epr_config = EPRConfig(**config_box.epr)
        epr = EPR(config=epr_config)
        epr_trajectory = epr.generate_synthetic_trajectory(
            n_agents=config_box.epr.generate.n_agents,
            start_date=start_time,
            end_date=end_time,
            tessellation=tessellation,
            starting_cells=None,
            random_state=random_seed
        )
        print("✅ EPR model trajectory generated successfully.")
        if config_box.epr.generate.save_output_trajectory:
            filename = f"{config_box.epr.generate.output_filename_prefix}_{run_datetime}.geojson"
            output_file_path = os.path.join(output_path, filename)
            epr_trajectory.to_file(output_file_path, driver='GeoJSON')
            print(f"✅ EPR trajectory saved to {output_file_path}")

    except ValidationError as e:
        print("❌ Error in EPR configuration:\n", e)
        exit(1)

if config_box.execution.run_ditras:
    if not "mc" in locals():
        print(
            "⚠️ Ditras model requires a fitted Markov Chain model. Please enable and run the Markov Chain model first.")
        exit(1)
    else:
        print("✅ Found fitted Markov Chain model. Proceeding with Ditras model.")
        try:
            epr_config = EPRConfig(**config_box.ditras)
            ditras = Ditras(config=epr_config, diary_generator=mc)
            ditras_trajectory = ditras.generate_synthetic_trajectory(
                n_agents=config_box.ditras.generate.n_agents,
                start_date=start_time,
                end_date=end_time,
                tessellation=tessellation,
                random_state=random_seed
            )
            print("✅ Ditras model trajectory generated successfully.")
            if config_box.ditras.generate.save_output_trajectory:
                filename = f"{config_box.ditras.generate.output_filename_prefix}_{run_datetime}.geojson"
                output_file_path = os.path.join(output_path, filename)
                ditras_trajectory.to_file(output_file_path, driver='GeoJSON')
                print(f"✅ Ditras trajectory saved to {output_file_path}")
        except ValidationError as e:
            print("❌ Error in Ditras configuration:\n", e)
            exit(1)

if config_box.execution.copy_config:
    output_filename = f"config_{run_datetime}.yaml"
    output_path = os.path.join(config_box.general.output_path, output_filename)
    print(f"\n 📋 Copied configuration file to: {output_path}")
    shutil.copy("config.yaml", output_path)
