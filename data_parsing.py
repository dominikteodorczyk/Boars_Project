"""
Data Parsing Script

This script processes CSV files using a JSON configuration file that specifies
the mapping between CSV files and the columns to retrieve.

The JSON configuration file must have the following structure:
    {
        "path/to/csv_file.csv": [
            "time_column",
            "agent_id_column",
            "longitude_column",
            "latitude_column"
        ],
        ...
    }

Each key is the path to a CSV file, and its value is a list specifying the
columns to retrieve. The list must include:
    - The time column (as the first entry).
    - The agent ID column (as the second entry).
    - The longitude and latitude columns (as the third and fourth entries).

The script loads the JSON configuration file, parses the data from
the specified CSV files, and processes it using the `parse_data` function
from `src.utils.parsers`.

Usage:
    - Place the configuration JSON path in COLUMNS_SOURCE.
    - Ensure all paths to CSV files are correctly specified and accessible.
    - Run this script to process the data based on the configuration.
"""

import json
import os
from src.utils.parsers import parse_data


JSON_CONFIG_FILE = ''  # Put the path to the JSON configuration file here
BREEDING_PERIODS = ''  # Put the path to the JSON file with breeding periods here

def main() -> None:
    """
    Main entry point of the script. This function orchestrates the loading of
    the JSON configuration file and processes the data using the `parse_data`
    function.

    The function performs the following steps:
    1. Loads the JSON configuration file, which specifies the input CSV files
       and other relevant parameters.
    2. Uses the `parse_data` function to process the CSV files listed in the
       configuration file, considering the breeding periods specified in
       the `BREEDING_PERIODS` JSON file.

    The JSON configuration file should follow a specific format that includes:
    - A list of CSV file paths to be processed.
    - A column layout where:
      - The first column contains the time data.
      - The second column contains the agent IDs.
      - The third and fourth columns represent the longitude and latitude
        coordinates of the agents.

    Args:
        None: This function does not take any arguments directly. It uses the
              global variables `JSON_CONFIG_FILE` and `BREEDING_PERIODS`
              for input data.

    Returns:
        None: This function does not return any value. It processes the data
              and performs side effects (e.g., saving the results).

    Example:
        Running this script with a valid JSON configuration and breeding
        periods file will:
        - Parse the CSV files listed in the configuration.
        - Process the data based on the breeding periods and other
            configurations.
        - Save the processed results to a specified location.
    """
    parse_data(json_source=JSON_CONFIG_FILE, periods=BREEDING_PERIODS)


if __name__ == "__main__":
    # Ensure the main function is called when the script is executed directly.
    main()


