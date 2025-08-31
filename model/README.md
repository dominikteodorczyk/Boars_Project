## Description

This project contains the implementation of mobility models for simulating user/animal movement patterns. It includes 
various techniques to generate synthetic trajectories based on real-world data for example: Markov Chains, 
EPR (Exploration and Preferential Return) model, and other statistical methods.

## Project Structure - Key Files and Directories
- `config.yaml`: Main configuration file. All data paths, simulation parameters, and settings for individual modules are defined here.
- `pyproject.toml`: File defining project dependencies.
- `src`: Source code directory containing all the modules and scripts.
    - `agent` - Contains classes and methods for store and manage agent data.
    - `EPR` - Contains the implementation of the EPR model for simulating movement patterns.
    - `markov_chain` - Contains the implementation of Markov Chain models for trajectory generation.
    - `gravity` - Contains the implementation of the Gravity model for simulating movement patterns.
- `utils` - Utility functions


## Configuration (`config.yaml`)

The `config.yaml` file is divided into several sections that control different aspects of the simulation.

### `general`
General settings and file paths.
- `file_path`: Path to the CSV input file with trajectory data.
- `tessellation_path`: Path to the GeoJSON file with tessellation (division of the area into smaller polygons).
- `output_path`: Directory where the simulation results will be saved.
- `start_simulation`: Date and time of the simulation start.
- `end_simulation`: Date and time of the simulation end.
- `random_seed`: Random seed to ensure repeatability of results.

### `execution`
Flags to control which processes to run.
- `run_markov`: `true` to run the Markov Chain model.
- `run_epr`: `true` to run the EPR model.
- `run_ditras`: `true` to run the Ditras model.
- `copy_config`: `true` to copy the configuration file to the output directory for archiving.

### `markov`
Parameters for a model based on Markov chains.
- `chain_length`: The length of the generated state chain.
- `time_slot`: The time interval used for data aggregation (e.g., “1h” for one hour).
- `label_column`: Name of the column in the input data that contains location labels.
- `generate`:
  - `enabled`: `true` to enable the generation of an abstract trajectory using this model.
  - `duration_hours`: Duration of the generated simulation in hours.

### `epr`
Parameters for the EPR (Exploration and Preferential Return) model.
- `rho`, `gamma`, `beta`, `tau`: Mathematical parameters of the EPR model.
- `min_waiting_time_minutes`: Minimum waiting time for an agent at a given location (in minutes).
- `tessellation_attractiveness_column`: Name of the column in the tessellation file that determines the attractiveness of the polygon (e.g., `population`).
- `generate`:
  - `n_agents`: Number of agents to simulate.
  - `save_output_trajectory`: `true` to save the generated trajectories.
  - `output_filename_prefix`: Prefix for output file names.

### `ditras`
Parameters for the Ditras model (a variation of the EPR model).
- `rho`, `gamma`, `beta`, `tau`: Mathematical parameters of the Ditras model.
- `min_waiting_time_minutes`: Minimum waiting time for an agent at a given location (in minutes).
- `tessellation_attractiveness_column`: Name of the column in the tessellation file that determines the attractiveness of the polygon.
- `simulation_with_attractiveness_raster`: `true` to use the attractiveness raster to update the attractiveness of the aggregation grid cells
- `attractiveness_raster_path`: Path to the raster file (e.g., GeoTIFF) defining the attractiveness of the terrain.
- `generate`:
  - `n_agents`: Number of agents to simulate.
  - `save_output_trajectory`: `true` to save the generated trajectories.
  - `output_filename_prefix`: Prefix for output file names.

## How to get started

This branch uses Poetry to manage dependencies and the environment.

This branch uses Poetry to manage dependencies and the environment.

1. **Installing Poetry:**

    If you do not have Poetry installed, follow the official documentation:
    https://python-poetry.org/docs/#installation

2. **Project configuration:**

    Before the first run, make sure that the `Boars_Project/model/config.yaml` file is configured correctly.

3. **Installing dependencies:**

    Navigate to the project directory and run:
    ```bash
    poetry install
    ```
4. Navigate to the project directory (`Boars_Project/model`) and run:
    ```bash
    poetry run python main.py
    ```
   
