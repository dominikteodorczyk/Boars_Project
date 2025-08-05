import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Pandas doesn't allow columns to be created via a new attribute name")

from humobi.structures.trajectory import TrajectoriesFrame
from model.src.markov_chain import MarkovChain, MarkovChainConfig
from model.src.EPR import EPR, EPRConfig, Ditras

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

tessellation_path = config_box.general.tessellation_path
tessellation = gpd.GeoDataFrame.from_file(tessellation_path)

start_time = pd.to_datetime(config_box.general.start_simulation)
end_time = pd.to_datetime(config_box.general.end_simulation)

try:
    markov_config = MarkovChainConfig(**config_box.markov)
except ValidationError as e:
    print("❌ Błąd walidacji konfiguracji:\n", e)
    exit(1)

mc = MarkovChain(config=markov_config)
mc.fit(trajectories_frame)
diary = mc.generate(48, pd.to_datetime(start_time, dayfirst=True), seed=42)

try:
    epr_config = EPRConfig(**config_box.epr)
except ValidationError as e:
    print("❌ Błąd walidacji konfiguracji:\n", e)
    exit(1)

epr = EPR(config=epr_config)
epr.generate_synthetic_trajectory(2, start_time, end_time, tessellation=tessellation, starting_cells=None,
                                  random_state=42)

ditras = Ditras(epr_config, mc)
ditras_gen = ditras.generate_synthetic_trajectory(2, start_time, end_time, tessellation=tessellation, random_state=42)

ditras_gen.to_file(
    os.path.join(config_box.general.output_path, fr"ditras_own_{datetime.now().strftime('%Y%m%d')}.geojson"),
    driver='GeoJSON')

output_filename = f"config_{datetime.now().strftime('%Y%m%d')}.yaml"
output_path = os.path.join(config_box.general.output_path, output_filename)
print(f"Copying config.yaml to {output_path}")
shutil.copy("config.yaml", output_path)
