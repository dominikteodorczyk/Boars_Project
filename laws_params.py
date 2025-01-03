import os
import glob
import logging
import pandas as pd
import numpy as np
from src.dataIO import DataIO, get_file_paths, create_output_directory
from src.laws import ScalingLawsCalc, DataSetStats
import traceback
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

INFOSTOP_DATA_DIR = ''
OUTPUT_DIR = 'scaling_laws_output'


def main():

    try:
        # Retrieve file paths from the data directory
        file_paths = get_file_paths(INFOSTOP_DATA_DIR)

        # Create output directory for results
        output_dir = create_output_directory(
            base_path=os.path.dirname(__file__),
            dir_name=OUTPUT_DIR
        )

        dataset_stats = DataSetStats(output_dir)

        # Process each file
        for parsed_file in file_paths:
            try:
                logging.info(f"Processing file: {parsed_file}")

                clean_data = DataIO.open_for_scaling_laws(parsed_file)
                data_name = DataIO.get_animal_name(parsed_file)

                sl_calc = ScalingLawsCalc(clean_data, data_name, output_dir, dataset_stats)
                sl_calc.process_file()
            except Exception as e:
                logging.warning(f'There is problem with {os.path.basename(parsed_file)}: {e}')

        dataset_stats.stats_set.to_csv('animal_stats.csv')

    except Exception as e:
        logging.error(
            f"An unexpected error occurred in the main workflow: {e}"
        )
        traceback.print_exc()



if __name__ == "__main__":
    main()
