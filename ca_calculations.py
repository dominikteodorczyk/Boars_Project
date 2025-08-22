"""
This module runs the calculation of the Ca coefficient (e.g., an association index
between animals) based on input data in CSV format. The data are loaded into a
pandas DataFrame, then passed to the TimeCa class for processing and computation.

The resulting Ca values are exported to a file named `animal_ca.csv` in the
current working directory.

"""

from src.contact import TimeCa
import pandas as pd
import os

PATH = ''  # Path to the input CSV file


def main():
    """
    Load input data, compute the Ca coefficient, and export the results to a CSV file.

    Steps:
        1. Read data from a CSV file specified by PATH into a pandas DataFrame.
        2. Create an instance of the TimeCa class.
        3. Pass the input data to the TimeCa instance.
        4. Compute the Ca coefficient using the `compute2` method.
        5. Save the results into `animal_ca.csv`.

    Raises:
        FileNotFoundError: If the file specified in PATH does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        Exception: For any other error that occurs during data processing.
    """
    data = pd.read_csv(PATH)
    ca = TimeCa()
    ca.input_data(data)
    result = ca.compute2()
    result.to_csv('animal_ca.csv')


if __name__ == "__main__":
    # Ensure the main function is executed only when the script is run directly.
    main()
