import pandas as pd
import os
import logging
from humobi.structures.trajectory import TrajectoriesFrame
from humobi.measures.individual import *
from humobi.tools.processing import *
from humobi.tools.user_statistics import *
from src.measures.stats import AnimalStatistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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
        self.animal = os.path.basename(csv_path).replace(".csv",'').replace("Trajectory_processed_","").replace("_"," ")
        self.file_name_to_write = os.path.basename(csv_path)
        self.raw_data = pd.read_csv(csv_path)
        self.raw_data["time"] = pd.to_datetime(self.raw_data["time"], unit="s")
        self.labels_preprocessed = None
        self.statistics = AnimalStatistics()

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
                "names": [
                    "num",
                    "labels",
                    "lat",
                    "lon",
                    "time",
                    "animal_id"
                    ],
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
                    label_df[
                        [
                            "user_id",
                            "labels",
                            "datetime",
                            "lat",
                            "lon"
                            ]
                        ],
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
                "names": [
                    "num",
                    "labels",
                    "lat",
                    "lon",
                    "time",
                    "animal_id"],
                "crs": 4326,
            },
        )

        compressed =  pd.DataFrame(
            start_end(mean_geometry_points).reset_index()[
                [
                    "user_id",
                    "datetime",
                    "labels",
                    "lat",
                    "lon",
                    "date",
                    "start",
                    "end"]
            ]
        )

        compressed[['user_id','labels','lat','lon','start','end']].to_csv(f'compressed_{self.file_name_to_write}')

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
                "names": [
                    "num",
                    "labels",
                    "lat",
                    "lon",
                    "time",
                    "animal_id"
                    ],
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
        self.data_to_use = TrajectoriesFrame(df_nonaf.loc[dis_loc[dis_loc > min_labels_no].index])

    def filter_q1(self):

        self._get_preprocessed_data()
        df_nonaf = self.labels_preprocessed[
            ~self.labels_preprocessed.isna().any(axis=1)
        ]
        dis_loc = num_of_distinct_locations(df_nonaf)
        q1 = np.quantile(dis_loc,0.25)

        if q1 < 3:
            q1 == 3

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
