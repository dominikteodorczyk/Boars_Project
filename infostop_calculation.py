"""
Script for processing parsed trajectory data and performing
Infostop analysis.

This script is designed to process trajectory data files,
prepare them for analysis using  he Infostop algorithm,
and store the results in an output directory. It consists
of the following steps:

1. Retrieve a list of file paths from a specified directory
    containing parsed data files.
2. Create an output directory to store the results.
3. Process each file individually by:
   - Cleaning and preparing the data.
   - Running the Infostop algorithm to compute stops and movements.
   - Storing the processed results.

Usage:
------
1. Define the directory path to the parsed data (PARSED_DATA_DIR).
2. Run the script to process all files in the given directory,
    generate outputs, and store results in an `infostop_output` folder.
3. The results of the Infostop analysis will be stored in the
    created output directory.

Example:
--------
python process_data.py
"""

import os
import traceback
import logging
from src.utils.dataIO import (
    DataPrepocessing,
    DataIO,
    get_file_paths,
    create_output_directory
)
from src.utils.istop import InfoStopData

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the directory containing parsed data files
PARSED_DATA_DIR = ''
OUTPUT_DIR_NAME = 'infostop_output'


def process_file(parsed_file:str, output_dir:str):
    """
    Processes a single parsed file to clean data, prepare it for
    infostop analysis, and generate outputs.

    Args:
        parsed_file (str): Path to the parsed data file.
        output_dir (str): Path to the output directory for storing results.

    Raises:
        Exception: Any errors during data preparation
            or processing will be raised.
    """
    try:
        # Prepare data for processing
        clean_data = DataIO.open_for_infostop(parsed_file)
        data_name = DataIO.get_animal_name(parsed_file)

        # Create and process an InfoStopData object
        infostop_object = InfoStopData(
            data=clean_data,
            data_name=data_name,
            output_dir=output_dir
        )
        infostop_object.calculate_all()

    except Exception as e:
        logging.error(f"Error processing file {parsed_file}: {e}")
        traceback.print_exc()


def main():
    """
    Main function to process all parsed data files, generate outputs,
    and handle errors.

    Workflow:
        1. Retrieve file paths from the parsed data directory.
        2. Create an output directory for infostop results.
        3. Process each file and handle any exceptions during the process.
    """
    try:
        # Retrieve file paths from the parsed data directory
        file_paths = get_file_paths(PARSED_DATA_DIR)

        # Create output directory for results
        output_dir = create_output_directory(
            base_path=os.path.dirname(__file__),
            dir_name=OUTPUT_DIR_NAME
        )

        # Process each parsed file
        for parsed_file in file_paths:
            logging.info(f"Processing file: {parsed_file}")
            process_file(parsed_file, output_dir)

    except Exception as e:
        logging.error(
            f"An unexpected error occurred in the main workflow: {e}"
        )
        traceback.print_exc()


if __name__ == "__main__":
    main()