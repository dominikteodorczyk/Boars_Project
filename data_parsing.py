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


JSON_CONFIG_FILE = '' ## put here path to JSON configuration file

def main() -> None:
    """
    Main function that loads a JSON configuration file and processes
    the data.

    The function performs the following steps:
    1. Loads a JSON file,
    2. Uses the `parse_data` function to process the CSV files
        listed in the JSON file.

    The JSON file should follow the specified format for the
        column list, where:
        - The first column is the time.
        - The second column is the agent ID.
        - The third and fourth columns are the longitude and latitude
            coordinates.

    Example:
        Assuming the JSON file contains valid data,
        running this script will parse the corresponding CSV files and save
        the processed results.

    """
    parse_data(json_source=JSON_CONFIG_FILE)


if __name__ == "__main__":

    main()


