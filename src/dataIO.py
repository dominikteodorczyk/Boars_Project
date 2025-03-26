"""
Module for handling data input/output operations related to trajectory
analysis.

This module provides functions and a class (`DataIO`) for managing file
paths, reading trajectory data from CSV files, processing them,
and extracting metadata like animal names.
"""

from numpy import size
import pandas as pd
import os
import logging
from humobi.structures.trajectory import TrajectoriesFrame
from humobi.measures.individual import *
from humobi.tools.processing import *
from humobi.tools.user_statistics import *
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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



class DataIO:
    """
    A class providing static methods for handling trajectory
    data input and processing.

    This class includes methods to open and preprocess
    trajectory data files, ensuring they are formatted correctly
    for further analysis.
    """

    @staticmethod
    def open_for_scaling_laws(csv_path: str) -> TrajectoriesFrame:
        """
        Opens a CSV file and converts it into a TrajectoriesFrame suitable
        for scaling law analysis.

        Args:
            csv_path (str): The path to the CSV file containing
                trajectory data.

        Returns:
            TrajectoriesFrame: A structured data frame with the
                loaded trajectory data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"The file at {csv_path} does not exist."
            )

        raw_data = pd.read_csv(csv_path)
        raw_data["time"] = pd.to_datetime(raw_data["time"], unit="s")

        return TrajectoriesFrame(
                    raw_data,
                    {
                        "names": ["num", "labels", "lat", "lon", "time", "animal_id"],
                        "crs": 4326,
                    },
                )


    @staticmethod
    def open_for_infostop(csv_path: str) -> TrajectoriesFrame:
        """
        Opens a CSV file and processes it for use in Infostop analysis.
        Filters out rows with missing values and converts coordinates
        to a different CRS.

        Args:
            csv_path (str): Path to the CSV file containing the raw data.

        Returns:
            TrajectoriesFrame: Processed trajectory data with valid
            coordinates and datetimes.

        Raises:
            FileNotFoundError: If the CSV file is not found at the given path.
            ValueError: If the CSV file contains invalid or malformed data.
            KeyError: If expected columns ('datetime', 'user_id', 'geometry',
                'lon', 'lat') are missing.
        """

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"The file at {csv_path} does not exist."
            )

        try:
            # Load data from the CSV file, create a TrajectoriesFrame and
            # filter the data
            raw_data = TrajectoriesFrame(
                pd.read_csv(csv_path), {"crs": 4326}
            )
            tframes = raw_data.reset_index()[
                ["datetime", "user_id", "geometry", "lon", "lat"]
            ]
            tframes.dropna(subset=["datetime", "lon", "lat"], inplace=True)
            clear_frame = TrajectoriesFrame(tframes)

            # Convert to the target CRS
            try:
                clear_frame = clear_frame.to_crs(
                    dest_crs=3857, cur_crs=4326
                ).copy()  # type: ignore
            except Exception as e:
                raise Exception(f"Error during CRS transformation: {e}")

            # Rename columns to match the required format
            clear_frame.columns = ['geometry','lat','lon']
            transformed_frame = clear_frame[["geometry", "lat", "lon"]]  # type: ignore

            # Remove duplicate rows
            final_frame = transformed_frame.drop_duplicates()

            return TrajectoriesFrame(final_frame)

        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing the CSV file: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")


    @staticmethod
    def get_animal_name(csv_path: str) -> str:
        """
        Extracts the animal's name from the file path by parsing
        the file name.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            str: Extracted animal name from the file name.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            ValueError: If the file name does not conform to the
                expected format.
        """
        # Check if the file exists at the provided path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"The file at {csv_path} does not exist."
            )

        try:
            # Extract the file name from the path
            file_name = os.path.basename(csv_path)

            # Ensure the file name contains the necessary part
            if not (
                file_name.startswith("Trajectory_processed_")
                or file_name.startswith("parsed_")
            ):
                raise ValueError(
                    f"The file name '{file_name}' "
                    f"does not match the expected format."
                )

            # Process the file name to extract the animal's name
            animal_name = (
                file_name.replace(  # Remove any leading or trailing spaces
                    ".csv", ""
                )  # Remove any trailing characters like .csv)
                .replace(
                    "Trajectory_processed_", ""
                )  # Remove the "Trajectory_processed_" prefix
                .replace("parsed_", "")  # Remove the "parsed" suffix if present
            )

            return animal_name

        except Exception as e:
            raise ValueError(
                f"Error extracting animal name from "
                f"file '{csv_path}': {e}"
            )

