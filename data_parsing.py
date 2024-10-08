"""
Data parsing script. As a configuration file it uses a json file
with the structure:
key = path to csv file,
values = list of 4 columns to retrieve where the first is the
    time column, the second is the agent id and then coordinates
"""


import json
import os
from src.utils.parsers import parse_data


def main() -> None:
    """
    Main function that loads a JSON configuration file and processes
    the data.

    The function performs the following steps:
    1. Loads a JSON file named 'cols.json', located in
        the 'constans' directory
        relative to the current working directory.
    2. Uses the `parse_data` function to process the CSV files
        listed in the JSON file.

    The JSON file should follow the specified format for the
        column list, where:
        - The first column is the time.
        - The second column is the agent ID.
        - The third and fourth columns are the longitude and latitude
            coordinates.

    Example:
        Assuming the JSON file at 'constans/cols.json' contains valid data,
        running this script will parse the corresponding CSV files and save
        the processed results.

    """
    parse_data(json_source='constans/cols.json')


if __name__ == "__main__":

    main()


