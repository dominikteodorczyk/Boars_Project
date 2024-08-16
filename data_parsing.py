"""
Data parsing script. As a configuration file it uses a json file
with the structure:
key = path to csv file,
values = list of 4 columns to retrieve where the first is the
    time column, the second is the agent id and then coordinates
"""


import json
import os
from src.utils.parsers import multi_raw_data_parser


def main() -> None:
    """
    Main function that loads a JSON configuration file and
    processes it using the multi_raw_data_parser.

    The function performs the following steps:
    1. Loads a JSON file named 'cols.json' located in the 'src/utils'
        directory relative to the current working directory.
    2. Passes the loaded data to the `multi_raw_data_parser`
        function for processing.

    Example:
        Assuming `cols.json` contains valid JSON data, running
        this script will parse the data using the `multi_raw_data_parser`.

    Returns:
        None
    """
    data_dict = json.load(
        open(
            os.path.join(
                os.getcwd(),
                'src',
                'utils',
                'cols.json'
                )
            )
        )

    multi_raw_data_parser(
        data_dict=data_dict
        )


if __name__ == "__main__":

    main()


