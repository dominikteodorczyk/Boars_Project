from datetime import timedelta
from numpy import size
import pandas as pd
import os
import logging
from humobi.structures.trajectory import TrajectoriesFrame
from src.measures.measures import Measures
from src.measures.stats import AnimalStatistics
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from humobi.measures.individual import *
from humobi.tools.processing import *
from humobi.tools.user_statistics import *
from constans import const

sns.set_style("whitegrid")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

class Stats:
    """
    A utility class for calculating various statistics and metrics
    from a TrajectoriesFrame.
    """

    @staticmethod
    def get_animals_no(data:TrajectoriesFrame) -> int:
        """
        Get the total number of unique animals in the dataset.

        Args:
            data (TrajectoriesFrame): Input data with animal trajectories.

        Returns:
            int: The number of unique animals.

        """
        return len(data.get_users())

    @staticmethod
    def get_period(data:TrajectoriesFrame) -> pd.Timedelta:
        """
        Get the total observation period of the dataset.

        Args:
            data (TrajectoriesFrame): Input data containing
                'start' and 'end' columns.

        Returns:
            pd.Timedelta: The duration between the earliest start
                and the latest end.

        Raises:
            KeyError: If columns 'start' or 'end' are missing
                in the dataset.
            ValueError: If columns 'start' or 'end' contain
                invalid datetime values.
        """
        try:
            min_start = pd.to_datetime(data["start"]).min()
            max_end = pd.to_datetime(data["end"]).max()
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {e}")
        return max_end - min_start

    @staticmethod
    def get_min_labels_no_after_filtration(
        data:TrajectoriesFrame
    ) -> pd.Series:
        """
        Get the users with the minimum number of unique labels
        after filtration.

        Args:
            data (TrajectoriesFrame): Input data with 'user_id'
                and 'labels' columns.

        Returns:
            pd.Series: Users with the minimum unique label count.
        """
        unique_label_counts = data.groupby('user_id')['labels'].nunique()
        min_unique_label_count = unique_label_counts.min()
        return unique_label_counts[
            unique_label_counts == min_unique_label_count
        ]

    @staticmethod
    def get_min_records_no_before_filtration(
        data:TrajectoriesFrame
    ) -> pd.Series:
        """
        Get the animals with the minimum number of records
        before filtration.

        Args:
            data (TrajectoriesFrame): Input data with 'animal_id'
                and 'time' columns.

        Returns:
            pd.Series: Animals with the minimum number of records.
        """
        records_counts = data.reset_index().groupby('user_id').datetime.count()
        min_label_count = records_counts.min()
        return records_counts[records_counts == min_label_count]

    @staticmethod
    def get_mean_periods(data:TrajectoriesFrame) -> pd.Timedelta:
        """
        Get the mean period between start and end times.

        Args:
            data (TrajectoriesFrame): Input data with 'start'
                and 'end' columns.

        Returns:
            float: The mean period in days.
        """
        return (data['end'] - data['start']).mean() # type: ignore

    @staticmethod
    def get_min_periods(data:TrajectoriesFrame) -> pd.Timedelta:
        """
        Get the minimum period between start and end times.

        Args:
            data (TrajectoriesFrame): Input data with 'start'
                and 'end' columns.

        Returns:
            float: The minimum period in days.
        """
        return (data['end'] - data['start']).min() # type: ignore

    @staticmethod
    def get_max_periods(data:TrajectoriesFrame) -> pd.Timedelta:
        """
        Get the maximum period between start and end times.

        Args:
            data (TrajectoriesFrame): Input data with 'start'
                and 'end' columns.

        Returns:
            float: The maximum period in days.
        """
        return (data['end'] - data['start']).max() # type: ignore

    @staticmethod
    def get_overall_area(data:TrajectoriesFrame) -> float:
        """
        Get the overall area covered by the trajectories.

        Args:
            data (TrajectoriesFrame): Input spatial data.

        Returns:
            float: The overall area in hectares.

        """
        convex_hull = data.unary_union.convex_hull
        return round(convex_hull.area / 10000,0)

    @staticmethod
    def get_mean_area(data:TrajectoriesFrame) -> float:
        """
        Get the mean area covered by trajectories per user.

        Args:
            data (TrajectoriesFrame): Input data.

        Returns:
            float: The mean area in hectares.
        """
        grouped = data.copy().groupby('user_id')
        areas = []
        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            areas.append(convex_hull.area / 10000)
        return round(sum(areas) / len(areas),0)

    @staticmethod
    def get_min_area(data:TrajectoriesFrame) -> float:
        """
        Get the min area covered by trajectories per user.

        Args:
            data (TrajectoriesFrame): Input data.

        Returns:
            float: The min area in hectares.
        """
        grouped = data.copy().groupby('user_id')
        areas = []
        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            areas.append(convex_hull.area / 10000)
        return round(min(areas),0)

    @staticmethod
    def get_max_area(data:TrajectoriesFrame) -> float:
        """
        Get the max area covered by trajectories per user.

        Args:
            data (TrajectoriesFrame): Input data.

        Returns:
            float: The max area in hectares.
        """
        grouped = data.copy().groupby('user_id')
        areas = []
        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            areas.append(convex_hull.area / 10000)
        return round(max(areas),0)


class DataSetStats:

    def __init__(self, output_dir) -> None:
        self.output_dir = output_dir
        self.record = {}
        self.stats_frame = pd.DataFrame(columns=[
                'animal',
                'animal_no',
                'animal_after_filtration',
                'time_period',
                'min_label_no',
                'min_records',
                'avg_duration',
                'min_duration',
                'max_duration',
                'overall_set_area',
                'average_set_area',
                'min_area',
                'max_area',
                'visitation_frequency',
                "visitation_frequency_params",
                "distinct_locations_over_time",
                "distinct_locations_over_time_params",
                "jump_lengths_distribution",
                "jump_lengths_distribution_params",
                "waiting_times",
                "waiting_times_params",
                "msd_curve",
                "msd_curve_params",
                "travel_times",
                "travel_times_params",
                "rog",
                "rog_params",
                "rog_over_time",
                "rog_over_time_params",
                "msd_distribution",
                "msd_distribution_params",
                "return_time_distribution",
                "return_time_distribution_params",
                "exploration_time",
                "exploration_time_params",
            ]
        )

    def add_data(self, data:dict) -> None:
        self.record.update(data)

    def add_record(self) -> None:
        self.stats_frame._append(self.record,ignore_index = True) # type: ignore
        self.record = {}


class Prepocessing:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_mean_points(data:TrajectoriesFrame) -> TrajectoriesFrame:
        basic_df = data.reset_index()
        geometry_df = pd.DataFrame()
        for an_id, values in tqdm(
            data.groupby(level=0), total=len(data.groupby(level=0))
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
                ) # type: ignore

        return TrajectoriesFrame(geometry_df.sort_values("datetime").drop_duplicates())

    @staticmethod
    def set_start_stop_time(data:TrajectoriesFrame) -> TrajectoriesFrame:
        compressed = pd.DataFrame(
            start_end(data).reset_index()[
                [
                    "user_id",
                    "datetime",
                    "labels",
                    "lat",
                    "lon",
                    "date",
                    "start",
                    "end"
                ]
            ]
        )
        return TrajectoriesFrame(
            compressed,
            {
                "names": ["num", "labels", "lat", "lon", "time", "animal_id"],
                "crs": const.ELLIPSOIDAL_CRS,
            },
        )

    @staticmethod
    def set_crs(
        data:TrajectoriesFrame,
        base_csr:int = const.ELLIPSOIDAL_CRS,
        target_crs:int=const.CARTESIAN_CRS
    ) -> TrajectoriesFrame:

        data.set_crs(base_csr)
        data.to_crs(target_crs)

        return data

    @staticmethod
    def filter_by_min_number(
        data:TrajectoriesFrame, min_labels_no:int = const.MIN_LABEL_NO
    ) -> TrajectoriesFrame:

        data_without_nans = data[data.isna().any(axis=1)]
        distinct_locations = num_of_distinct_locations(data_without_nans)

        return TrajectoriesFrame(
            data_without_nans.loc[
                distinct_locations[
                    distinct_locations > min_labels_no
                ].index
            ]
        )


    @staticmethod
    def filter_by_quartiles(
        data: TrajectoriesFrame, quartile: float = const.QUARTILE
    ) -> TrajectoriesFrame:

        allowed_quartiles = {0.25, 0.5, 0.75}
        if quartile not in allowed_quartiles:
            raise ValueError(f"Invalid quartile value: {quartile}. "
                             f"Allowed values are {allowed_quartiles}."
            )
        else:
            data_without_nans = data[~data.isna().any(axis=1)]
            distinct_locations = num_of_distinct_locations(data_without_nans)
            quartile_value = np.quantile(distinct_locations, quartile)

            if quartile_value < const.MIN_QUARTILE_VALUE:
                quartile_value = const.MIN_QUARTILE_VALUE

            return TrajectoriesFrame(
                data_without_nans.loc[
                    distinct_locations[
                        distinct_locations > quartile_value
                    ].index]
            )



class Laws:

    def __init__(self, pdf_object: FPDF, stats_dict:dict, output_path: str) -> None:
        self.pdf_object = pdf_object
        self.output_path = output_path




class ScalingLawsCalc:

    def __init__(
            self,
            data:TrajectoriesFrame,
            data_name:str,
            output_dir:str,
            stats_frame:DataSetStats
        ) -> None:
        self.data = data
        self.animal_name = data_name
        self.output_dir = output_dir
        self.output_dir_animal = os.path.join(output_dir, data_name)
        self.pdf = FPDF()
        self.stats_frame = stats_frame

        try:
            os.mkdir(self.output_dir_animal)
        except FileExistsError:
            raise FileExistsError(
                f"The directory {self.output_dir_animal} already exists. "
                f"Please remove it or specify a different name."
            )

        self.pdf.add_page()
        self.pdf.set_font("Arial", size=9)
        self.pdf.cell(200, 10, text=f"{self.animal_name}", ln=True, align="C")

    def _preprocess_data(self) -> tuple:
        preproc = Prepocessing()
        stats = Stats()

        mean_points_values = preproc.get_mean_points(self.data)
        compressed_points = preproc.set_start_stop_time(mean_points_values)

        converted_to_cartesian = preproc.set_crs(compressed_points)
        filtered_animals = preproc.filter_by_quartiles(converted_to_cartesian)

        print('RAW ANIMAL NO:',stats.get_animals_no(self.data))
        print('FILTRED ANIMAL NO:',stats.get_animals_no(filtered_animals))

        print('RAW ANIMAL PERIOD:',stats.get_period(filtered_animals))
        print('FILTRED ANIMAL PERIOD:',stats.get_period(filtered_animals))

        print('MIN RECORDS NO BEF FILTRATION :',stats.get_min_records_no_before_filtration(self.data))
        print('MIN LABELS NO AFTER FILTRATION :',stats.get_min_labels_no_after_filtration(filtered_animals))

        #FIXME: choose data for compressed csv and next step of calculations
        return compressed_points, filtered_animals


    def process_file(self) -> None:

        compressed_points, filtered_animals = self._preprocess_data()





