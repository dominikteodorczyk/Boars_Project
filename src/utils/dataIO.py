from numpy import size
import pandas as pd
import os
import logging
from humobi.structures.trajectory import TrajectoriesFrame
from humobi.measures.individual import *
from humobi.tools.processing import *
from humobi.tools.user_statistics import *
from src.measures.stats import AnimalStatistics
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataIO:

    @staticmethod
    def open_for_scaling_laws(csv_path):
        pass

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
            raise FileNotFoundError(f"The file at {csv_path} does not exist.")

        try:
            # Load data from the CSV file, create a TrajectoriesFrame and filter the data
            raw_data = TrajectoriesFrame(pd.read_csv(csv_path), {"crs": 4326})
            tframes = raw_data.reset_index()[
                ["datetime", "user_id", "geometry", "lon", "lat"]
            ]
            tframes.dropna(subset=["datetime", "lon", "lat"], inplace=True)
            clear_frame = TrajectoriesFrame(tframes)

            # Convert to the target CRS
            try:
                clear_frame = clear_frame.to_crs(dest_crs=3857, cur_crs=4326).copy()  # type: ignore
            except Exception as e:
                raise Exception(f"Error during CRS transformation: {e}")

            # Rename columns to match the required format
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
            raise FileNotFoundError(f"The file at {csv_path} does not exist.")

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
                f"Error extracting animal name from " f"file '{csv_path}': {e}"
            )


class DataPrepocessing:
    """
    A class for preprocessing animal trajectory data from a CSV file.
    """

    def __init__(self, csv_path) -> None:
        """
        Initializes the DataPrepocessing class by loading data from
        a CSV file, setting up the animal name, converting time column to
        datetime, and initializing statistics.

        Parameters:
        ----------
        csv_path : str
            Path to the CSV file containing raw animal trajectory data.
        """
        self.csv_path = csv_path
        self.raw_data = pd.read_csv(csv_path)
        self.file_name_to_write = os.path.basename(csv_path)

        # SCALING LAWS atributs
        self.animal = None
        self.labels_preprocessed = None
        self.statistics = None
        self.data_to_use = None

        # INFOSTOP atributs
        self.parsed_data_to_use = None

    def scaling_laws_prepare(self):

        self.animal = (
            os.path.basename(self.csv_path)
            .replace(".csv", "")
            .replace("Trajectory_processed_", "")
            .replace("_", " ")
        )
        self.raw_data["time"] = pd.to_datetime(self.raw_data["time"], unit="s")
        self.labels_preprocessed = None
        self.statistics = AnimalStatistics()

    def infostop_data_prepare(self):

        tframes = TrajectoriesFrame(self.raw_data, {"crs": 4326})
        tframes = tframes.reset_index()[
            ["datetime", "user_id", "geometry", "lon", "lat"]
        ]
        tframes = tframes[
            tframes["datetime"].notna()
            & tframes["lon"].notna()
            & tframes["lat"].notna()
        ]
        tframes = TrajectoriesFrame(tframes)
        tframes = tframes.to_crs(dest_crs=3857, cur_crs=4326)
        tframes.columns = ["geometry", "lat", "lon"]
        tframes = tframes.drop_duplicates()

        self.infostop_stats = None

        return TrajectoriesFrame(tframes), self.file_name_to_write.replace(
            "parsed_", ""
        ).replace(".csv", "")

    def _get_mean_points(self):
        """
        Calculates mean latitude and longitude points for each unique label of
        each user.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame with mean points for each label of each user, sorted
            by datetime and with duplicates removed.
        """

        self.statistics.get_min_records_no_before_filtration(self.raw_data)
        _df = TrajectoriesFrame(
            self.raw_data,
            {
                "names": ["num", "labels", "lat", "lon", "time", "animal_id"],
                "crs": 4326,
            },
        )

        basic_df = _df.reset_index()
        geometry_df = pd.DataFrame()

        for an_id, values in tqdm(
            _df.groupby(level=0), total=len(_df.groupby(level=0))
        ):
            filtered_df = basic_df[basic_df["user_id"] == an_id]
            lables_pack = sorted(filtered_df["labels"].unique())

            for lbl in lables_pack:
                label_df = filtered_df[filtered_df["labels"] == lbl]
                label_df["lat"] = label_df["lat"].mean()
                label_df["lon"] = label_df["lon"].mean()
                label_df = label_df.reset_index()

                geometry_df = geometry_df._append(
                    label_df[["user_id", "labels", "datetime", "lat", "lon"]],
                    ignore_index=True,
                )

        compressed = geometry_df.sort_values("datetime").drop_duplicates()

        return compressed

    def _set_start_stop_time(self):
        """
        Sets the start and stop times for the trajectory data.

        Returns:
        -------
        pandas.DataFrame
            A DataFrame containing user_id, datetime, labels, lat, lon, date,
            start, and end times.
        """
        mean_geometry_points = TrajectoriesFrame(
            self._get_mean_points(),
            {
                "names": ["num", "labels", "lat", "lon", "time", "animal_id"],
                "crs": 4326,
            },
        )

        compressed = pd.DataFrame(
            start_end(mean_geometry_points).reset_index()[
                ["user_id", "datetime", "labels", "lat", "lon", "date", "start", "end"]
            ]
        )

        compressed[["user_id", "labels", "lat", "lon", "start", "end"]].to_csv(
            f"compressed_{self.file_name_to_write}"
        )

        return compressed

    def _get_preprocessed_data(self):
        """
        Preprocesses the raw data by setting start and stop times, transforming
        coordinates, and updating statistics.
        """
        ready_data = self._set_start_stop_time()
        self.labels_preprocessed = TrajectoriesFrame(
            ready_data,
            {
                "names": ["num", "labels", "lat", "lon", "time", "animal_id"],
                "crs": 4326,
            },
        )
        self.labels_preprocessed = self.labels_preprocessed.set_crs(4326)
        self.labels_preprocessed = self.labels_preprocessed.to_crs(3857)

        self.statistics.get_raw_animals_no(self.labels_preprocessed)
        self.statistics.get_raw_period(self.labels_preprocessed)

    def filter_above_number(self, min_labels_no=3):
        """
        Filters the preprocessed data by removing rows with missing values
        and selecting users with more than one distinct location.
        """

        self._get_preprocessed_data()
        df_nonaf = self.labels_preprocessed[
            ~self.labels_preprocessed.isna().any(axis=1)
        ]
        dis_loc = num_of_distinct_locations(df_nonaf)
        self.data_to_use = TrajectoriesFrame(
            df_nonaf.loc[dis_loc[dis_loc > min_labels_no].index]
        )

    def filter_q1(self):

        self._get_preprocessed_data()
        df_nonaf = self.labels_preprocessed[
            ~self.labels_preprocessed.isna().any(axis=1)
        ]
        dis_loc = num_of_distinct_locations(df_nonaf)
        q1 = np.quantile(dis_loc, 0.25)

        if q1 < 3:
            q1 = 3

        self.data_to_use = TrajectoriesFrame(df_nonaf.loc[dis_loc[dis_loc > q1].index])

    def filter_data(self):

        self.filter_q1()

        self.statistics.get_raw_filtered_animals_no(self.data_to_use)
        self.statistics.get_filtered_period(self.data_to_use)
        self.statistics.get_min_labels_no_after_filtration(self.data_to_use)

        self.statistics.get_mean_periods(self.data_to_use)
        self.statistics.get_min_periods(self.data_to_use)
        self.statistics.get_max_periods(self.data_to_use)

        self.statistics.get_overall_area(self.data_to_use)
        self.statistics.get_mean_area(self.data_to_use)
        self.statistics.get_min_area(self.data_to_use)
        self.statistics.get_max_area(self.data_to_use)

        return self.data_to_use
