"""
Script for processing animal trajectory data and performing scaling
law calculations.

This script:
- Loads trajectory data files from a specified directory.
- Creates an output directory for storing analysis results.
- Processes each file to clean and structure the data.
- Applies scaling law calculations to the data.
- Saves the processed statistics to a CSV file.

If an error occurs during processing, it is logged, and the script continues
with the next file.
"""


import os
import glob
import logging
import pandas as pd
import numpy as np
from src.dataIO import DataIO, get_file_paths, create_output_directory
from src.laws import ScalingLawsCalc, DataSetStats
import traceback
import logging
import matplotlib
matplotlib.use('Agg')

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

INFOSTOP_DATA_DIR = ""
OUTPUT_DIR = "scaling_laws_output"


def main():
    """
    Main function of the script processing animal trajectory data.

    The script performs the following steps:
    1. Retrieves file paths from the input data directory.
    2. Creates an output directory for analysis results.
    3. Initializes an object to store dataset statistics.
    4. Iterates through files, loads and processes the data, and then performs
        scaling law calculations.
    5. Exports dataset statistics to a CSV file.

    In case of errors, logs them and continues processing the next files.
    """
    try:
        # Retrieve file paths from the data directory
        file_paths = get_file_paths(INFOSTOP_DATA_DIR)

        # Create output directory for results
        output_dir = create_output_directory(
            base_path=os.path.dirname(__file__), dir_name=OUTPUT_DIR
        )

        dataset_stats = DataSetStats(output_dir)

        # Process each file
        for parsed_file in file_paths:
            try:
                logging.info(f"Processing file: {parsed_file}")

                clean_data = DataIO.open_for_scaling_laws(parsed_file)
                data_name = DataIO.get_animal_name(parsed_file)

                sl_calc = ScalingLawsCalc(
                    clean_data, data_name, output_dir, dataset_stats
                )
                sl_calc.process_file()
            except Exception as e:
                logging.warning(
                    f"There is problem with {os.path.basename(parsed_file)}: {e}"
                )

        dataset_stats.stats_set.to_csv("animal_stats.csv")

    except Exception as e:
        logging.error(f"An unexpected error occurred in the main workflow: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
