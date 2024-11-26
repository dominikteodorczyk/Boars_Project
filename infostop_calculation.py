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
from src.utils.dataIO import DataPrepocessing
from src.utils.istop import InfoStopData

# Define the directory containing parsed data files
PARSED_DATA_DIR = ''
OUTPUT_DIR_NAME = ''


def get_file_paths(directory:str) -> list:
    """
    Retrieves a list of file paths from a given directory.

    Args:
        directory (str): The path to the directory containing the files.

    Returns:
        list: A list of full file paths to all files in the directory.
    """
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file))
    ]


def create_output_directory(base_path:str, dir_name: str) -> str:
    """
    Creates an output directory if it does not already exist.

    Args:
        base_path (str): The base path where the directory should be created.
        dir_name (str): The name of the output directory.

    Returns:
        str: The full path to the created directory.

    Raises:
        OSError: If the directory cannot be created due to system errors.
    """
    output_path = os.path.join(base_path, dir_name)
    if not os.path.exists(output_path):
        try:
            os.mkdir(output_path)
        except OSError as e:
            raise OSError(f"Failed to create output "
                          f"directory at {output_path}: {e}")
    return output_path


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
        data = DataPrepocessing(parsed_file)
        clean_data, data_name = data.infostop_data_prepare()

        # Create and process an InfoStopData object
        infostop_object = InfoStopData(
            data=clean_data,
            data_name=data_name,
            output_dir=output_dir
        )
        infostop_object.calculate_all()

    except Exception as e:
        print(f"Error processing file {parsed_file}: {e}")
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
            print(f"Processing file: {parsed_file}")
            process_file(parsed_file, output_dir)

    except Exception as e:
        print(f"An unexpected error occurred in the main workflow: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()