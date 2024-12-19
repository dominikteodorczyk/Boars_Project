"""
Module for data analysis and labeling using the Infostop algorithm.

This module provides functionality for analyzing trajectory data
and identifying "stops" and "moves" using the Infostop algorithm.
It includes multiple classes and methods designed to handle the
following tasks:

- Filtering and processing trajectory data.
- Applying the Infostop algorithm to detect stops based on location and time.
- Computing and visualizing optimal parameters for the Infostop algorithm.
- Generating reports and plots for analysis.
"""


import os
import logging
from io import BytesIO
from multiprocessing import Pool
from itertools import product
import pandas as pd
from infostop import Infostop
import numpy as np
from tqdm import tqdm
from fpdf import FPDF
from humobi.structures.trajectory import TrajectoriesFrame
from humobi.tools.user_statistics import (
    fraction_of_empty_records,
    count_records,
    count_records_per_time_frame,
    user_trajectories_duration,
    consecutive_record,
)
from constans import const

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import warnings
warnings.filterwarnings(
    "ignore",
    message="Pandas doesn't allow columns to be created via a new attribute name"
    )

class DataAnalystInfostop:
    """
    A class responsible for generating data analysis reports in PDF format.

    Parameters
    ----------
    pdf_object : FPDF
        PDF instance to which the analysis results will be added.
    """

    def __init__(self, pdf_object: FPDF, output_path: str) -> None:
        """
        Initializes the DataAnalystInfostop with a given PDF instance.

        Parameters
        ----------
        pdf_object : FPDF
            PDF instance where analysis results will be written.
        """
        self.pdf_object = pdf_object
        self.output_path = output_path

    def generate_raport(
            self,
            data: TrajectoriesFrame,
            data_analyst_no: int
    ) -> None:
        """
        Generates a data analysis report, adding results and visualizations
        to the PDF.

        Parameters
        ----------
        data : TrajectoriesFrame
            Input data for analysis.
        data_analyst_no : int
            Analysis identifier used in titles and filenames.

        Notes
        -----
        Executes a data analysis pipeline, generating various statistical
        metrics and plots, which are sequentially added to the PDF document.
        """
        self._add_analyst_title(data_analyst_no)
        self._add_basic_statistics(data, data_analyst_no)
        self._add_total_records_statistics(data, data_analyst_no)
        self._add_records_per_time_frame(data, data_analyst_no, "1D")
        self.pdf_object.add_page()

        self._add_records_per_time_frame(data, data_analyst_no, "1H")
        self._add_trajctories_duration(data, data_analyst_no, "1H")
        self._add_no_of_consecutive_records(data, data_analyst_no, "1H")
        self.pdf_object.add_page()

        self._add_no_of_consecutive_records(data, data_analyst_no, "1D")
        self._add_average_temporal_resolution(data, data_analyst_no)
        self.pdf_object.add_page()

    def _add_pdf_cell(self, txt_to_add: str) -> None:
        """
        Adds a single line of text to the PDF.

        Parameters
        ----------
        txt_to_add : str
            Text to be added to the PDF document.
        """
        self.pdf_object.cell(200, 5, text=txt_to_add, ln=True, align="L")

    def _add_pdf_plot(
        self,
        plot_obj: BytesIO,
        image_width: int,
        image_height: int,
        x_position: int = 10,
    ) -> None:
        """
        Adds a plot to the PDF at the current cursor position.

        Parameters
        ----------
        plot_obj : BytesIO
            Plot image to add to the PDF.
        image_width : int
            Width of the plot in the PDF.
        image_height : int
            Height of the plot in the PDF.
        x_position : int, optional
            X-coordinate position of the plot, by default 10.
        """
        y_position = self.pdf_object.get_y()
        self.pdf_object.image(
            plot_obj,
            x=x_position,
            y=y_position,
            w=image_width,
            h=image_height
        )
        self.pdf_object.set_y(y_position + image_height + 10)
        plot_obj.close()

    def _plot_fraction_of_empty_records(
        self, frac: pd.Series, data_analyst_no: int
    ) -> BytesIO:
        """
        Generates a plot showing the fraction of empty records by threshold.

        Parameters
        ----------
        frac : Series
            Series containing fraction of empty records.
        data_analyst_no : int
            Analysis identifier for the plot filename.

        Returns
        -------
        BytesIO
            In-memory image buffer of the generated plot.
        """
        thresholds = [round(i, 2) for i in np.arange(0.01, 1.01, 0.01)]
        num_of_complete = {
            threshold: (frac <= threshold).sum() for threshold in thresholds
        }

        buffer = BytesIO()
        plt.figure(figsize=(10, 6))
        plt.plot(num_of_complete.keys(), num_of_complete.values(), lw=3)
        plt.title("Counts_frac vs. Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Counts_frac")
        plt.grid(True)
        plt.xlim(0, 1)
        plt.savefig(
            os.path.join(
                self.output_path,
                f"Data Analysis {data_analyst_no}: "
                f"Counts frac vs Threshold.png",
            )
        )
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        return buffer

    def _plot_distribution(
        self, data: pd.Series, plot_name: str, bins: int = 10
    ) -> BytesIO:
        """
        Generates a distribution plot of the given data.

        Parameters
        ----------
        data : Series or array-like
            Data to plot.
        plot_name : str
            Filename for saving the plot.
        bins : int, optional
            Number of bins for the histogram, by default 10.

        Returns
        -------
        BytesIO
            In-memory image buffer of the generated distribution plot.
        """
        buffer = BytesIO()
        sns.displot(data, kde=True, bins=bins)  # type: ignore
        plt.savefig(os.path.join(self.output_path, plot_name))
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        return buffer

    def _add_analyst_title(self, data_analyst_no: int) -> None:
        """
        Adds the title of the analysis section to the PDF.

        Parameters
        ----------
        data_analyst_no : int
            Analysis identifier used in the title.
        """
        self._add_pdf_cell(f"Data Analysis: {data_analyst_no}")

    def _add_basic_statistics(
        self, data: TrajectoriesFrame, data_analyst_no: int
    ) -> None:
        """
        Adds basic statistics to the PDF, including the number of animals
        and fraction of empty records.

        Parameters
        ----------
        data : TrajectoriesFrame
            Input data for analysis.
        data_analyst_no : int
            Analysis identifier.
        """
        number_of_animals = len(data.get_users())
        self._add_pdf_cell(f"Number of animals: {number_of_animals}")

        frac = fraction_of_empty_records(data, resolution="1H")
        frac_median = frac.median()
        frac_mean = frac.mean()

        plot_obj = self._plot_fraction_of_empty_records(
            frac=frac, data_analyst_no=data_analyst_no
        )
        self._add_pdf_cell(f"Fraction of empty records (median): "
                           f"{frac_median}")
        self._add_pdf_cell(f"Fraction of empty records (mean): {frac_mean}")

        self._add_pdf_plot(plot_obj, 100, 60)

    def _add_total_records_statistics(self, data, data_analyst_no) -> None:
        """
        Adds total record statistics to the PDF, including median
        and mean values.

        Parameters
        ----------
        data : TrajectoriesFrame
            Input data for analysis.
        data_analyst_no : int
            Analysis identifier.
        """
        count = count_records(data)
        count_median = count.median()
        count_mean = count.mean()

        self._add_pdf_cell(f"Total number of records (median): {count_median}")
        self._add_pdf_cell(
            f"Total number of records (mean): {count_mean}",
        )

        plot_obj = self._plot_distribution(
            data=count,
            plot_name=f"Data Analysis {data_analyst_no}: "
            f"Total numbers of records.png",
        )

        self._add_pdf_plot(plot_obj, 60, 60)

    def _add_records_per_time_frame(
            self, data, data_analyst_no, resolution
        ) -> None:
        """
        Adds statistics on records per specified time frame to the PDF.

        Parameters
        ----------
        data : TrajectoriesFrame
            Input data for analysis.
        data_analyst_no : int
            Analysis identifier.
        resolution : str
            Time resolution for analysis (e.g., '1D', '1H').
        """
        count_per_time_frame = count_records_per_time_frame(
            data,
            resolution=resolution
        )
        count_per_time_frame_median = count_per_time_frame.median()
        count_per_time_frame_mean = count_per_time_frame.mean()

        self._add_pdf_cell(
            f"Total number of records per time frame {resolution} "
            f"(median): {count_per_time_frame_median}"
        )
        self._add_pdf_cell(
            f"Total number of records per time frame {resolution} "
            f"(mean): {count_per_time_frame_mean}"
        )

        records_per_animal = count_per_time_frame.groupby(level=0).median().median()
        self._add_pdf_cell(
            f"Average records per animal per {resolution}: "
            f"{records_per_animal}"
        )

        plot_obj = self._plot_distribution(
            data=count_per_time_frame.groupby(level=0).median(),
            plot_name=f"Data Analysis {data_analyst_no}: "
            f"Count per {resolution}.png",
        )

        self._add_pdf_plot(plot_obj, 60, 60)

    def _add_trajctories_duration(
            self, data, data_analyst_no, resolution
        ) -> None:
        """
        Adds trajectory duration statistics to the PDF.

        Parameters
        ----------
        data : TrajectoriesFrame
            Input data for analysis.
        data_analyst_no : int
            Analysis identifier.
        resolution : str
            Time resolution for trajectory duration (e.g., '1H').
        """
        trajectories_duration_1H = user_trajectories_duration(
            data, resolution=resolution, count_empty=False
        )

        self._add_pdf_cell(
            f"Trajectories duration ({resolution} median): "
            f"{trajectories_duration_1H.median()}"
        )

        plot_obj = self._plot_distribution(
            data=trajectories_duration_1H,
            plot_name=f"Data Analysis {data_analyst_no}: "
            f"Trajectories duration ({resolution}) distribution.png",
        )
        self._add_pdf_plot(plot_obj, 60, 60)

    def _add_no_of_consecutive_records(
            self, data, data_analyst_no, resolution
        ) -> None:
        """
        Adds statistics on consecutive records to the PDF.

        Parameters
        ----------
        data : TrajectoriesFrame
            Input data for analysis.
        data_analyst_no : int
            Analysis identifier.
        resolution : str
            Time resolution for consecutive records analysis (e.g., '1H').
        """
        consecutive_1h = consecutive_record(data, resolution=resolution)
        consecutive_1h_median = consecutive_1h.median()

        self._add_pdf_cell(
            f"Median of consecutive records ({resolution}):"
            f" {consecutive_1h_median}"
        )

        plot_obj = self._plot_distribution(
            data=consecutive_1h,
            plot_name=f"Data Analysis {data_analyst_no}: "
            f"Distribution of consecutive records ({resolution}).png",
        )
        self._add_pdf_plot(plot_obj, 60, 60)

    def _add_average_temporal_resolution(
            self, data, data_analyst_no
        ) -> None:
        """
        Adds average temporal resolution statistics to the PDF.

        Parameters
        ----------
        data : TrajectoriesFrame
            Input data for analysis, which includes datetime information
            required to calculate temporal resolution.
        data_analyst_no : int
            Analysis identifier used in plot filenames.
        """
        if len(data.get_users()) != 1:
            temporal_df = (
                data.reset_index(level=1)
                .groupby(level=0)
                .apply(lambda x: x.datetime - x.datetime.shift())
            )
        else:
            data_reset = data.reset_index()
            temporal_df = data_reset['datetime'] - data_reset['datetime'].shift()

        temporal_df_median = temporal_df.median()
        temp_res_animal = temporal_df.groupby(level=0).median()
        in_minutes = temporal_df[~temporal_df.isna()].dt.total_seconds() / 60
        in_minutes_filtred = in_minutes[in_minutes > 0]

        self._add_pdf_cell(f"Median temporal resolution: "
                           f"{temporal_df_median}"
                           )

        self._add_pdf_cell(f"Max temporal resolution: "
                           f"{in_minutes_filtred.max()}"
                           )

        plot_obj = self._plot_distribution(
            data=in_minutes_filtred,
            plot_name=f"Data Analysis {data_analyst_no}: "
            f"The distribution of average temporal resolution.png",
        )
        self._add_pdf_plot(plot_obj, 60, 60)


class DataFilter:
    """
    A class responsible for filtering, processing, and sorting
    animal trajectory data.

    Attributes:
        pdf_object (FPDF): PDF object used for documenting
            the data processing steps.
        allowed_minutes (list): List of allowed temporal
            resolutions in minutes.
        day_window (int): Time window in days for selecting
            the best periods of data.
    """

    def __init__(self, pdf_object: FPDF) -> None:
        """
        Initializes the DataFilter object with a PDF for logging
        and default settings.

        Args:
            pdf_object (FPDF): The PDF object for adding logging
                information.

        Attributes:
            allowed_minutes (list): A list of allowed temporal
                resolutions (in minutes).
            day_window (int): The number of days considered for
                the time window.
        """
        self.pdf_object = pdf_object
        self.allowed_minutes = const.ALLOWED_MINUTES
        self.day_window = const.DAYS_WINDOW

    def _add_pdf_cell(self, txt_to_add: str) -> None:
        """
        Adds a single line of text to the PDF.

        Args:
            txt_to_add (str): The text to be added to the PDF document.
        """
        self.pdf_object.cell(200, 5, text=txt_to_add, ln=True, align="L")

    def _match_timedelta(self, timedelta: pd.Series) -> tuple:
        """
        Matches a given timedelta to the closest allowed
        temporal resolution.

        Args:
            timedelta (pd.Timedelta): The timedelta to match.

        Returns:
            tuple: A tuple containing the string representation
                of the closest temporal resolution (e.g., '10min')
                and its integer value in minutes.
        """
        try:
            closest_minute = min(
                self.allowed_minutes, key=lambda x: abs(
                    pd.Timedelta(minutes=x) - timedelta
                )  # type: ignore
            )  # type: ignore
            return f"{closest_minute}min", closest_minute
        except Exception as e:
            raise ValueError(f"Error matching timedelta: "
                             f"{timedelta}. Details: {e}"
                        )

    def _select_time_window(self, closest_minute: int) -> int:
        """
        Calculates the number of records required for a rolling window
        based on the closest temporal resolution.

        Args:
            closest_minute (int): The closest temporal resolution in minutes.

        Returns:
            int: The number of records required for a rolling window.
        """
        try:
            per_hour = 60 / closest_minute
            per_day = per_hour * 24
            return int(self.day_window * per_day)
        except Exception as e:
            raise ValueError(
                f"Error calculating time window for resolution "
                f"{closest_minute}: {e}"
            )

    def _extract_best_coverage(
        self, data: pd.DataFrame, data_coverage: pd.Series
    ) -> pd.DataFrame:
        """
        Extracts data with the best coverage based on a rolling window.

        Args:
            data (pd.DataFrame): The input data containing user trajectories.
            data_coverage (pd.Series): The data coverage calculated over
                a rolling window.

        Returns:
            pd.DataFrame: The extracted data with the best coverage.
        """
        try:
            extracted = []
            for an_id, an_val in data_coverage.groupby(level=0):
                if an_val.isna().all():
                    continue
                max_date = an_val.idxmax()[1]  # type: ignore
                till_EOD = (
                    pd.to_datetime(
                        max_date.date() + pd.Timedelta(days=1)  # type: ignore
                    )
                    - max_date
                )
                max_date += till_EOD
                an_val = an_val.reset_index()
                min_date = max_date - pd.Timedelta(days=self.day_window)
                cur_df = data.loc[an_id].reset_index()
                selected = cur_df[
                    (cur_df["datetime"] <= max_date) &
                    (cur_df["datetime"] >= min_date)
                ].copy()
                selected["user_id"] = an_id
                selected = selected.set_index(["user_id", "datetime"])
                extracted.append(selected)

            return pd.concat(extracted)
        except Exception as e:
            raise ValueError(f"Error extracting best coverage: {e}")

    def _convert_to_unix(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a datetime column to UNIX timestamps.

        Args:
            group (pd.DataFrame): The input DataFrame with a 'time' column.

        Returns:
            pd.DataFrame: The DataFrame with the 'time' column
                converted to UNIX timestamps.
        """
        try:
            group["time"] = group["time"].apply(
                lambda x: int(pd.to_datetime(x).timestamp())
            )
            return group
        except Exception as e:
            raise ValueError(f"Error converting datetime "
                             f"to UNIX timestamps: {e}"
                        )

    def select_best_period(self, data: pd.DataFrame) -> TrajectoriesFrame:
        """
        Selects the best periods of data to maximize coverage.

        Args:
            data (pd.DataFrame): The input data containing user trajectories.

        Returns:
            TrajectoriesFrame: A TrajectoriesFrame object containing
                the selected data.

        Raises:
            ValueError: If there is an error during data selection.
        """
        try:
            if len(data.get_users()) != 1:
                temporal_df = (
                    data.reset_index(level=1)
                    .groupby(level=0)
                    .apply(lambda x: x.datetime - x.datetime.shift())
                )
            else:
                data_reset = data.reset_index()
                temporal_df = data_reset['datetime'] - data_reset['datetime'].shift()

            temporal_df_median = temporal_df.median()
            avg_temp_res_str, avg_temp_res_int = self._match_timedelta(
                timedelta= temporal_df_median
            )
            resampled = (
                data.groupby(level=0)
                .resample(avg_temp_res_str, level=1)
                .count()
                .iloc[:, 0]
            )

            check_time = self._select_time_window(avg_temp_res_int)
            resampled[resampled > 1] = 1
            data_coverage = resampled.groupby(level=0).apply(
                lambda x: x.rolling(check_time).sum() / check_time
            )
            extracted = self._extract_best_coverage(data, data_coverage)

            self._add_pdf_cell("Selecting best periods - "
                               "maximizing data coverage"
                            )
            self._add_pdf_cell(f"Avg temporal resolution: "
                               f"{temporal_df_median}"
                            )
            self._add_pdf_cell(f"Resample value (str): {avg_temp_res_str}")
            self._add_pdf_cell(f"Resample value (int): {avg_temp_res_int}")
            self._add_pdf_cell(
                f"Selected time window for {self.day_window} "
                f"days (timecheck): {check_time}"
            )
            self.pdf_object.add_page()

            return TrajectoriesFrame(extracted)
        except Exception as e:
            raise ValueError(f"Error selecting best period: {e}")

    def filter_data(self, data: pd.DataFrame) -> TrajectoriesFrame:
        """
        Filters the data based on various user statistics.

        Args:
            data (pd.DataFrame): The input data containing user trajectories.

        Returns:
            TrajectoriesFrame: A TrajectoriesFrame object containing
                the filtered data.

        Raises:
            ValueError: If there is an error during filtering.
        """
        try:
            temporal_df = (
                data.reset_index(level=1)
                .groupby(level=0)
                .apply(lambda x: x.datetime - x.datetime.shift())
            )
            temp_res_animal_ex = temporal_df.groupby(level=0).median()

            # Fraction of missing records < 0.6
            frac_filter = fraction_of_empty_records(data, const.RESOLUTION_OF_FRACTION_OF_MISSING_VALUES)
            level1 = set(frac_filter[frac_filter < const.FRACTION_OF_MISSING_VALUES].index)

            # More than 20 days of data
            traj_dur_filter = user_trajectories_duration(data, const.RESOLUTION_OF_USER_TRAJECTORIES_DURATION)
            level2 = set(traj_dur_filter[traj_dur_filter > const.USER_TRAJECTORIES_DURATION].index)

            level3 = set(
                temp_res_animal_ex[temp_res_animal_ex <= const.TEMPORAL_RESOLUTION_EXTRACTION].index
            )

            # User filtration with ULOC method
            selection_lvl2 = level1.intersection(level2)
            selection_lvl3 = selection_lvl2.intersection(level3)
            filtered_data = data.uloc(list(selection_lvl3))  # type: ignore

            self._add_pdf_cell("Filtration with user statistics")
            self._add_pdf_cell(f"Number of animals on level 1: {len(level1)}")
            self._add_pdf_cell(f"Number of animals on level 2: {len(level2)}")
            self._add_pdf_cell(f"Number of animals on level 3: {len(level3)}")
            self.pdf_object.add_page()

            return TrajectoriesFrame(filtered_data)
        except Exception as e:
            raise ValueError(f"Error filtering data: {e}")

    def sort_data(self, data: TrajectoriesFrame) -> pd.DataFrame:
        """
        Sorts and prepares trajectory data for infostop processing.

        Args:
            data (TrajectoriesFrame): The input trajectory data.

        Returns:
            pd.DataFrame: A sorted and prepared DataFrame with columns
                ['user_id', 'time', 'latitude', 'longitude'].

        Raises:
            ValueError: If there is an error during sorting or preparation.
        """
        try:
            data_sorted = data.sort_index(level=[0, 1])
            data_sorted = data_sorted.to_crs(dest_crs=const.ELLIPSOIDAL_CRS, cur_crs=const.CARTESIAN_CRS)  # type: ignore
            df1 = data_sorted.reset_index()
            data_prepared = df1.reset_index(drop=True)[
                ["user_id", "datetime", "lon", "lat"]
            ]
            data_prepared.columns = [
                "user_id",
                "time",
                "latitude",
                "longitude"
            ]
            data_prepared = self._convert_to_unix(data_prepared)
            return data_prepared.groupby("user_id")  # type: ignore
        except Exception as e:
            raise ValueError(f"Error sorting data: {e}")


class LabelsCalc:
    """
    A class for processing and labeling movement data using
    the Infostop algorithm. Includes functionality for parameter optimization,
    trajectory segmentation, and result visualization.

    Attributes
    ----------
    pdf_object : FPDF
        A PDF object for documenting results and calculations.
    output_path : str
        The directory where plots and other outputs will be saved.
    """

    def __init__(self, pdf_object: FPDF, output_path: str) -> None:
        """
        Initialize the LabelsCalc instance.

        Parameters
        ----------
        pdf_object : FPDF
            PDF object for logging and documentation.
        output_path : str
            Path to save generated plots and outputs.
        """
        self.pdf_object = pdf_object
        self.output_path = output_path

    def _add_pdf_cell(self, txt_to_add: str) -> None:
        """
        Add a single line of text to the PDF.

        Parameters
        ----------
        txt_to_add : str
            Text to be added to the PDF document.
        """
        self.pdf_object.cell(200, 5, text=txt_to_add, ln=True, align="L")

    def _add_pdf_plot(
        self,
        plot_obj: BytesIO,
        image_width: int,
        image_height: int,
        x_position: int = 10,
    ) -> None:
        """
        Add a plot to the PDF.

        Parameters
        ----------
        plot_obj : BytesIO
            A BytesIO object containing the plot image.
        image_width : int
            Width of the plot in the PDF.
        image_height : int
            Height of the plot in the PDF.
        x_position : int, optional
            X-coordinate position of the plot, by default 10.
        """
        try:
            y_position = self.pdf_object.get_y()
            self.pdf_object.image(
                plot_obj,
                x=x_position,
                y=y_position,
                w=image_width,
                h=image_height
            )
            self.pdf_object.set_y(y_position + image_height + 10)
            plot_obj.close()
        except Exception as e:
            raise RuntimeError(f"Failed to add plot to PDF: {e}")

    def _compute_intervals(
        self, labels: list, times: np.ndarray, max_time_between: int = const.MAX_TIME_BETWEEN
    ) -> list:
        """
        Compute stop and move intervals based on labels and timestamps.

        Parameters
        ----------
        labels : np.ndarray
            Array of labels assigned to data points (e.g., stops or moves).
        times : np.ndarray
            Array of timestamps corresponding to the labels.
        max_time_between : int, optional
            Maximum allowable time (in seconds) between consecutive points,
            by default 86400.

        Returns
        -------
        list : A list of intervals with columns: label, start_time, end_time,
            latitude, longitude.
        """
        trajectory = np.hstack([labels.reshape(-1, 1), times.reshape(-1, 3)])  # type: ignore
        final_trajectory = []

        loc_prev, lat, lon, t_start = trajectory[0]
        t_end = t_start

        for loc, lat, lon, time in trajectory[1:]:
            if (loc == loc_prev) and (time - t_end) < max_time_between:
                t_end = time
            else:
                final_trajectory.append([loc_prev, lat, lon, t_start, t_end])
                t_start = time
                t_end = time
            loc_prev = loc
        if loc_prev == -1:
            final_trajectory.append([loc_prev, lat, lon, t_start, t_end])
        return final_trajectory

    def _process_combination(self, args: list) -> dict:
        """
        Process a single combination of parameters and compute trajectories.

        Parameters
        ----------
        args : Tuple
            A tuple containing user_id, group, r1, r2, and min_staying_time.

        Returns
        -------
        dict : Results containing user ID, trajectories, and parameter values.
        """
        user_id, group, r1, r2, min_staying_time = args

        try:
            group = group.sort_values("time")
            data = group[["latitude", "longitude", "time"]].values
            model = Infostop(
                r1=r1,
                r2=r2,
                label_singleton=False,
                min_staying_time=min_staying_time,
                max_time_between=const.MAX_TIME_BETWEEN,
                min_size=2,
            )

            labels = model.fit_predict(data)
            trajectory = self._compute_intervals(labels, data)
            trajectory = pd.DataFrame(
                trajectory, columns=["labels", "lat", "lon", "start", "end"]
            )
            trajectory = trajectory[trajectory.labels != -1]

            total_stops = len(np.unique(labels))
            results = {
                "animal_id": user_id,
                "Trajectory": trajectory,
                "Total_stops": total_stops,
                "R1": r1,
                "R2": r2,
                "Tmin": min_staying_time,
            }
            return results

        except Exception as e:
            raise RuntimeError(f"Error processing user {args[0]}: {e}")

    def _calc_params_matrix_parallel(
            self, data: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Calculate the parameter sensitivity matrix using parallel processing.

        Parameters
        ----------
        data : pd.DataFrame
            Input trajectory data grouped by user ID.

        Returns
        -------
        pd.DataFrame
            A DataFrame with sensitivity analysis results.
        """
        try:
            rs1 = np.logspace(1, 2, 20, base=50)
            rs2 = np.logspace(1, 2, 20, base=50)
            min_staying_times = np.logspace(
                np.log10(600),
                np.log10(7200),
                num=20
            )

            tasks = []
            for user_id, group in tqdm(data, total=len(data)):  # type: ignore
                for r1, r2, min_staying_time in product(
                    rs1,
                    rs2,
                    min_staying_times
                ):
                    tasks.append((user_id, group, r1, r2, min_staying_time))

            with Pool() as pool:
                results = list(
                    tqdm(
                        pool.imap(
                            self._process_combination,
                            tasks
                        ),
                        total=len(tasks)
                    )
                )

            return pd.DataFrame([res for res in results if res is not None])
        except Exception as e:
            raise RuntimeError(f"Error calculating parameter matrix in "
                               f"parallel: {e}")

    def _calc_params_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the parameter sensitivity matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Input trajectory data grouped by user ID.

        Returns
        -------
        pd.DataFrame
            A DataFrame with sensitivity analysis results.
        """
        rs1 = np.logspace(1, 2, 20, base=50)
        rs2 = np.logspace(1, 2, 20, base=50)
        min_staying_times = np.logspace(np.log10(600), np.log10(7200), num=20)

        dfs_list = []
        for user_id, group in tqdm(data, total=len(data)):  # type: ignore
            group = group.sort_values("time")
            data = group[["latitude", "longitude", "time"]].values
            for r1 in rs1:
                for r2 in rs2:
                    for min_staying_time in min_staying_times:
                        model = Infostop(
                            r1=r1,
                            r2=r2,
                            label_singleton=False,
                            min_staying_time=min_staying_time,
                            max_time_between=const.MAX_TIME_BETWEEN,
                            min_size=2,
                        )
                        try:
                            labels = model.fit_predict(data)
                            trajectory = self._compute_intervals(labels, data)  # type: ignore
                            trajectory = pd.DataFrame(
                                trajectory,
                                columns=[
                                    "labels", "lat", "lon", "start", "end"
                                ],
                            )
                            trajectory = trajectory[trajectory.labels != -1]

                            total_stops = len(np.unique(labels))
                            results = {
                                "animal_id": user_id,
                                "Trajectory": trajectory,
                                "Total_stops": total_stops,
                                "R1": r1,
                                "R2": r2,
                                "Tmin": min_staying_time,
                            }
                            dfs_list.append(results)
                        except Exception as e:
                            print(f"Error processing user {user_id}: {e}")

        return pd.DataFrame(dfs_list)

    def _plot_param(
        self,
        param: str,
        stabilization_point_index: float,
        x: np.ndarray,
        y: np.ndarray,
        dy_dx: np.ndarray,
    ) -> BytesIO:
        """
        Create and return a plot visualizing parameter sensitivity.

        Parameters
        ----------
        param : str
            The parameter being analyzed.
        stabilization_point_index : float
            The selected stabilization point for the parameter.
        x : np.ndarray
            X-axis values (parameter values).
        y : np.ndarray
            Y-axis values (total stops).
        dy_dx : np.ndarray
            Change in total stops with respect to the parameter.

        Returns
        -------
        BytesIO
            A BytesIO object containing the plot image.
        """
        fig, ax1 = plt.subplots(figsize=(25, 20))
        ax1.plot(x, y, "o", label="Original data", linestyle="-", linewidth=2)
        ax1.axvline(
            stabilization_point_index,
            color="orange",
            linestyle="-",
            label=f"Stable {param} = {stabilization_point_index:.0f}",
        )
        ax1.set_xlabel(param)
        ax1.set_ylabel("Total Stops")
        ax1.legend(loc="lower right")

        ax2 = ax1.twinx()
        ax2.plot(
            x, dy_dx, label="Change", color="red", linewidth=2, linestyle="-"
        )
        ax2.axhline(0, color="red", linestyle="--", linewidth=1)
        ax2.legend(loc="upper right")

        plt.title(f"Median Total Stops by {param}")

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        return buffer

    def _choose_param_value(self, data: pd.DataFrame, param: str) -> float:
        """
        Choose the optimal value for a given parameter based on
        the sensitivity matrix.

        The method identifies a stabilization point for the parameter,
        where the total stops stabilize with minimal changes in the
        corresponding value.

        Parameters
        ----------
        data : pd.DataFrame
            The data containing the results of parameter sensitivity.
        param : str
            The parameter to optimize. Should be one of ["R1", "Tmin"].

        Returns
        -------
        float
            The selected optimal parameter value based on
            the stabilization point.
        """

        if param == "R1":
            data = data[(data["R1"] >= const.R1_MIN) & (data["R1"] <= const.R1_MAX)]
        if param == "Tmin":
            data = data[(data["Tmin"] >= const.TMIN_MIN) & (data["Tmin"] <= const.TMIN_MAX)]

        result_med = data.groupby(["animal_id", param]).median()
        result_med_agg = result_med.groupby(
            level=param
        )["Total_stops"].median()

        x = result_med_agg.index
        y = result_med_agg.values

        dy_dx = np.diff(y)  # type: ignore
        dy_dx = np.insert(dy_dx, 0, 0)

        dxdy = pd.DataFrame(
            {"de_value": np.abs(dy_dx)}
        ).sort_values(by="de_value")

        stabilization_x = x[dxdy.index[:const.BEST_POINT_NO]]
        filtred = data[data[param].isin(stabilization_x)]
        suma_total_stops = filtred.groupby(param)["Total_stops"].sum()
        stabilization_point_index = float(suma_total_stops.idxmax())

        plot_obj = self._plot_param(
            param, stabilization_point_index, x, y, dy_dx  # type: ignore
        )

        self._add_pdf_cell(f"{param} value: {stabilization_point_index}")
        self._add_pdf_plot(plot_obj, 200, 150)

        return stabilization_point_index

    def _choose_best_params(
        self, sensitivity_matrix: pd.DataFrame
    ) -> tuple[float, float, float]:
        """
        Choose the best parameter values for R1, R2, and Tmin based on
        the sensitivity matrix.

        The method sequentially selects the best values for R1, R2,
        and Tmin based on their stabilization point and the effect
        on the total stops.

        Parameters
        ----------
        sensitivity_matrix : pd.DataFrame
            The sensitivity matrix with parameters (R1, R2, Tmin)
            and their respective total stops.

        Returns
        -------
        Tuple[float, float, float]
            A tuple containing the selected values for R1, R2, and Tmin.
        """
        try:
            r1 = self._choose_param_value(sensitivity_matrix, "R1")
            r2 = self._choose_param_value(sensitivity_matrix, "R2")
            self.pdf_object.add_page()  # Start a new page for the next section of raport
            Tmin = self._choose_param_value(sensitivity_matrix, "Tmin")

            return r1, r2, Tmin
        except Exception as e:
            raise RuntimeError(f"Error selecting best parameters: {e}")

    def _calc_labels(self, data, r1, r2, Tmin):
        results = []
        for user_id, group in tqdm(data, total=len(data)):
            group = group.sort_values("time")
            data = group[["latitude", "longitude", "time"]].values
            model = Infostop(
                r1=r1,
                r2=r2,
                label_singleton=False,
                min_staying_time=Tmin,
                max_time_between=const.MAX_TIME_BETWEEN,
                min_size=2,
            )
            labels = model.fit_predict(data)
            trajectory = np.hstack(
                [labels.reshape(-1, 1), data.reshape(-1, 3)]  # type: ignore
            )
            trajectory = pd.DataFrame(
                trajectory, columns=["labels", "lat", "lon", "time"]
            )
            trajectory = trajectory[trajectory.labels != -1]
            trajectory["animal_id"] = user_id

            results.append(
                pd.DataFrame(
                    trajectory,
                    columns=["animal_id", "labels", "lat", "lon", "time"]
                )
            )
        return pd.concat(results, ignore_index=True)

    def calculate_infostop(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform Infostop calculation, including sensitivity analysis
        and trajectory labeling.

        Parameters
        ----------
        data : pd.DataFrame
            Input data containing user trajectories.

        Returns
        -------
        pd.DataFrame
            Final labeled trajectories.
        """
        try:
            self._add_pdf_cell("Infostop calculations")
            sensitivity_matrix = self._calc_params_matrix_parallel(data)
            r1, r2, Tmin = self._choose_best_params(sensitivity_matrix)
            final_data = self._calc_labels(data, r1, r2, Tmin)
            self._add_pdf_cell(
                f"Final number of animals: "
                f"{final_data['animal_id'].nunique()}"
            )
            return final_data
        except Exception as e:
            raise RuntimeError(f"Error during Infostop calculation: {e}")


class InfoStopData:
    """
    A class responsible for analyzing animal data and generating reports in
    PDF and CSV formats.

    Attributes:
        clean_data (TrajectoriesFrame): Raw input data related to animals.
        animal_name (str): The name of the analyzed animal.
        output_dir (str): Path to the main output directory.
        output_dir_animal (str): Path to the output directory for the
            specific animal.
        pdf (FPDF): FPDF object used for generating PDF reports.
    """

    def __init__(
        self, data: TrajectoriesFrame, data_name: str, output_dir: str
    ) -> None:
        """
        Initializes the InfoStopData object, creates an output directory,
        and sets up a PDF report.

        Args:
            data (TrajectoriesFrame): The input data for analysis.
            data_name (str): The name of the analyzed animal.
            output_dir (str): Path to the main output directory.

        Raises:
            FileExistsError: If the output directory for the animal
                already exists.
        """
        self.clean_data = data
        self.animal_name = data_name
        self.output_dir = output_dir
        self.output_dir_animal = os.path.join(output_dir, data_name)
        self.pdf = FPDF()

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

    def calculate_all(self) -> None:
        """
        Performs a complete analysis of the data, generates reports
        in PDF format, and saves the results to a CSV file.

        Process:
            1. Generates an initial report.
            2. Selects the best period of data.
            3. Filters the data.
            4. Sorts the data.
            5. Processes the data using the infostop module.
            6. Exports results to CSV and PDF files.

        Raises:
            FileNotFoundError: If a required file is missing during
                the process.
            ValueError: If the data contains invalid values.
            Exception: For any unexpected errors that occur during
                the process.
        """
        raport = DataAnalystInfostop(self.pdf, self.output_dir_animal)
        filter = DataFilter(self.pdf)
        labels_calculator = LabelsCalc(self.pdf, self.output_dir_animal)

        try:
            # Step 1: Generate the initial report
            raport.generate_raport(data=self.clean_data, data_analyst_no=1)

            # Step 2: Select the best period of data
            extracted_data = filter.select_best_period(data=self.clean_data)
            raport.generate_raport(data=extracted_data, data_analyst_no=2)

            # Step 3: Filter the data
            filtred_data = filter.filter_data(data=extracted_data)

            if len(filtred_data) !=0:
                raport.generate_raport(data=filtred_data, data_analyst_no=3)

                # Step 4: Sort the data
                sorted_data = filter.sort_data(filtred_data)

                # Step 5: Process data with the infostop module
                trajectory_processed_data = labels_calculator.calculate_infostop(
                    sorted_data
                )

                # Save results to CSV
                csv_path = os.path.join(
                    self.output_dir,
                    f"Trajectory_processed_{self.animal_name}.csv"
                )
                trajectory_processed_data.to_csv(csv_path)
            else:
                logging.warning('No animals after filtration')

            # Save report to PDF
            pdf_path = os.path.join(
                self.output_dir_animal, f"{self.animal_name}.pdf"
            )
            self.pdf.output(pdf_path)

        except FileNotFoundError as e:
            print(f"File error: {e}")
            raise
        except ValueError as e:
            print(f"Data value error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
        finally:
            # Ensure the PDF is saved even if an error occurs
            pdf_path = os.path.join(
                self.output_dir_animal, f"{self.animal_name}.pdf"
            )
            self.pdf.output(pdf_path)
