## Description

This project aims to compare different models for generating synthetic trajectories with real data. It implements and evaluates three models: Exploration and Preferential Return (EPR), Random Walk, and Lévy Flight. The main goal is to evaluate how well each synthetic model reproduces the spatiotemporal properties of the original trajectories using metrics such as Earth Mover's Distance (EMD).

## Project Structure
The project is organized in a modular way to separate the logic of data processing, model simulation, and result analysis.

- `comparator.py`: The main script that orchestrates the entire process, from loading data to running simulations and generating comparison charts.
- `config.yaml`: Configuration file for managing input/output paths and model parameters.
- `trajectory_simulator.py`: Class responsible for initializing and running various trajectory generation models.
- `data_loader.py`: Responsible for loading and preprocessing raw trajectory data.
- `file_manager.py`: Manages file operations such as creating output directories.
- `geo_processor.py` & `trajectory_processor.py`: A set of classes for handling geospatial operations such as grid creation, cell ID assignment, temporal resampling, and other data processing tasks.
- `epr_trajectory.py`: Implementation of the EPR model, including parameter estimation based on input data.
- `random_walk_trajectory.py`: Implementation of the Random Walk model.
- `levy_flight_trajectory.py`: Implementation of the Lévy Flight model.
- `utils.py`: Contains helper functions for calculating metrics (e.g., `compute_emd`) and generating graphs.
- `logger.py`: A simple logger class for logging information about the process.

## Key Files

- `comparator.py`: The main model comparison file. It reads the configuration, processes the input files one by one, runs simulations for each model, and generates comparative results.
- `config.yaml`: Central location for defining paths to input data, output directory, and parameters such as quartile for filtering or resampling frequency.
- `epr_trajectory.py`: Contains logic specific to the EPR model. The most important part is the `param_estimate` method, which estimates the `rho` and `gamma` parameters based on actual data, and the `find_best_fit` method, which fits the distribution to the waiting times.
- `utils.py`: Implements key comparative metrics. The `compute_emd` function calculates the EMD distance between the spatial distributions of the original and synthetic trajectories for each hour.

## How to get started

This branch uses Poetry to manage dependencies and the environment.
1. **Installing Poetry:**

    If you do not have Poetry installed, follow the official documentation:
    https://python-poetry.org/docs/#installation

2. **Project configuration:**

    Before the first run, make sure that the `compare_models/src/config.yaml` file is configured correctly, especially the `input_dir` and `output_dir` paths.

3. **Installing dependencies:**

    Navigate to the project directory and run:
    ```bash
    poetry install
    ```
   
4. **Running the comparison:**

    To run the model comparison, execute:
    ```bash
    poetry run python compare_models/src/comparator.py
    ```

    The script will process all files from `input_dir`, and the results for each file will be saved in a separate subdirectory in `output_dir`.

## Important Information: 
### Modification of `scikit-mobility` for waiting times in EPR:

The standard implementation of `DensityEPR` in `scikit-mobility` uses a built-in waiting time model. This project uses a custom model based on the `distfit` library to better fit empirical data. For this to work, you need to modify your local installation of `scikit-mobility`.

Below is a description of how to modify the `DensityEPR` class to accept its own waiting time model (`wt_model`).

1. **Find the `epr.py` file:**

      This file is located in your `scikit-mobility` installation, usually under a path similar to:
       `.../site-packages/skmob/models/epr.py`

2. **Modify `__init__` in the `DensityEPR` class:**

    Add a new parameter `wt_model` to the `__init__` method:

    **Original code:**

        class DensityEPR(EPR):
            def __init__(self, name='Density EPR model', rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20, wt_model=None):
                super().__init__(rho=rho, gamma=gamma, beta=beta, tau=tau, min_wait_time_minutes=min_wait_time_minutes)
                self._name = name
                self.wt_model = wt_model
    **Modified code:**

        class DensityEPR(EPR):
            def __init__(self, name='Density EPR model', rho=0.6, gamma=0.21, beta=0.8, tau=17, min_wait_time_minutes=20, wt_model=None):
                super().__init__(rho=rho, gamma=gamma, beta=beta, tau=tau, min_wait_time_minutes=min_wait_time_minutes)
                self._name = name
                self.wt_model = wt_model

3. **Modify the `_choose_waiting_time` method in `EPR` class:**
    Replace the existing logic to use the custom `wt_model` if provided:

    **Original code:**

        def _choose_waiting_time(self):
             time_to_wait = self._time_generator()
             return time_to_wait

    **Modified code:**

        def _choose_waiting_time(self):
            wt = self.wt_model.generate(1)[0] / 3600
            return wt

After saving these changes, the `DensityEPR` class will be able to accept a fitted `distfit` object as `wt_model` and use it to generate waiting times, allowing for greater flexibility and better model fit to the data.