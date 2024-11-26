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
        records_counts = data.groupby('animal_id').time.count()
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

    def process_file(self) -> None:
        pass



