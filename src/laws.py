"""
Module for data processing, statistical analysis,
and model fitting using Flexation and related methods.

This module includes functionality for handling data related
to animal movement, applying various filtering and preprocessing
techniques, and fitting statistical models to segmented data based
on flexation points.
"""


import os
import logging
import scipy
import scipy.stats
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from numpy import ndarray, std
import pandas as pd


import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from humobi.structures.trajectory import TrajectoriesFrame
from humobi.measures.individual import (
    visitation_frequency,
    jump_lengths,
    distinct_locations_over_time,
    radius_of_gyration,
    mean_square_displacement,
    num_of_distinct_locations,
)
from humobi.tools.processing import (
    rowwise_average,
    convert_to_distribution,
    start_end,
    groupwise_expansion,
)
from constans import const
import scipy.stats as scp_stats
from distfit import distfit
from tqdm import tqdm
from io import BytesIO
from math import log
from scipy.stats import wasserstein_distance
import ruptures as rpt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import powerlaw


sns.set_style("whitegrid")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Curves:
    """
    A class containing various static methods to model different
    mathematical curves. These methods take input values and parameters
    to compute specific mathematical transformations.
    """

    @staticmethod
    def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Computes a linear transformation: y = a*x.

        Parameters:
        x (array-like): Input values.
        a (float): Slope coefficient.
        b (float): Scaling coefficient.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a * x


    @staticmethod
    def expon(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Computes an exponential curve: y = a * x^b.

        Parameters:
        x (array-like): Input values.
        a (float): Base scaling coefficient.
        b (float): Exponent.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a * np.power(x, b)


    @staticmethod
    def expon_neg(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Computes a negative exponential curve: y = a * x^(-b).

        Parameters:
        x (array-like): Input values.
        a (float): Scaling coefficient.
        b (float): Negative exponent.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a * pow(x, -b)


    @staticmethod
    def euler(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Computes an Euler's exponential curve: y = a * e^(b * x).

        Parameters:
        x (array-like): Input values.
        a (float): Scaling coefficient.
        b (float): Exponent coefficient.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a * np.exp(b * x)


    @staticmethod
    def power(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Computes a power curve: y = a * b^x.

        Parameters:
        x (array-like): Input values.
        a (float): Scaling coefficient.
        b (float): Base.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a * pow(b, x)


    @staticmethod
    def power_neg(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Computes a negative power curve: y = a * b^(-x).

        Parameters:
        x (array-like): Input values.
        a (float): Scaling coefficient.
        b (float): Base.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a * pow(b, -x)


    @staticmethod
    def logar(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Computes a logarithmic curve: y = a + b * log(x).

        Parameters:
        x (array-like): Input values.
        a (float): Constant offset.
        b (float): Logarithmic scaling factor.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a + b * np.log(x)


    @staticmethod
    def cubic(
        x: np.ndarray, a: float, b: float, c: float, d: float
    ) -> np.ndarray:
        """
        Computes a cubic curve: y = a*x^3 + b*x^2 + c*x + d.

        Parameters:
        x (array-like): Input values.
        a, b, c, d (float): Coefficients for cubic, quadratic,
            linear, and constant terms.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a * x**3 + b * x**2 + c * x + d


    @staticmethod
    def sigmoid(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Computes a sigmoid curve: y = 1 / (1 + exp(a*x)).

        Parameters:
        x (array-like): Input values.
        a (float): Scaling factor.
        b (float): Offset factor.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return 1 / (1 + np.exp(a * x))


    @staticmethod
    def quad(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """
        Computes a quadratic curve: y = a*x^2 + b*x + c.

        Parameters:
        x (array-like): Input values.
        a (float): Coefficient for the quadratic term.
        b (float): Coefficient for the linear term.
        c (float): Constant term.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a * x**2 + b * x + c


    @staticmethod
    def four(
        x: np.ndarray, a: float, b: float, c: float, d: float, e: float
    ) -> np.ndarray:
        """
        Computes a quartic curve: y = a*x^4 + b*x^3 + c*x^2 + d*x + e.

        Parameters:
        x (array-like): Input values.
        a, b, c, d, e (float): Coefficients for quartic, cubic,
        quadratic, linear, and constant terms.

        Returns:
        array-like: Transformed values.
        """

        x = x.astype(float)
        return a * x**4 + b * x**3 + c * x**2 + d * x + e


    @staticmethod
    def zipf(x: np.ndarray, a: float, b: float) -> np.ndarray:
        """
        Computes a Zipf curve: y = 1 / (x + a)^b.

        Parameters:
        x (array-like): Input values.
        a (float): Offset to prevent division by zero.
        b (float): Exponent for scaling.

        Returns:
        array-like: Transformed values.
        """

        return 1 / (x + a) ** b


    @staticmethod
    def power_law(n, A, B):
        """
        Power-law function used to model the number of unique places
        visited over time.
        """

        return A * n**B


class DistributionFitingTools:
    """
    A class containing tools for fitting distributions and models to data.
    """

    def __init__(self) -> None:
        self.curves = Curves()


    def _fit_distribution(
        self, data: np.ndarray, distribution: scipy.stats.rv_continuous
    ) -> float:
        """
        Fits a given distribution to the data

        Parameters:
        ----------
        data : numpy.ndarray
            The data to which the distribution is fitted.
        distribution : scipy.stats.rv_continuous
            The distribution to fit to the data.

        Returns:
        -------
        float
        """

        params = distribution.fit(data)
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        pdf_values = distribution.pdf(data, loc=loc, scale=scale, *arg)
        log_likelihood = np.sum(np.log(pdf_values))
        num_params = len(params)
        aic = 2 * len(params) - 2 * log_likelihood
        aicc = aic + (2 * num_params * (num_params + 1)) / (
            len(data) - num_params - 1
        )  # AICc correction

        return aicc


    def _calculate_akaike_weights(self, aic_values: list) -> np.ndarray:
        """
        Calculates Akaike weights from AIC values.

        Parameters:
        ----------
        aic_values : list of float
            The list of AIC values for different models.

        Returns:
        -------
        numpy.ndarray
            The Akaike weights corresponding to each AIC value.
        """

        delta_aic = aic_values - np.min(aic_values)
        exp_term = np.exp(-0.5 * delta_aic)
        weights = exp_term / np.sum(exp_term)

        return weights


    def _multiple_distributions(self, data: np.ndarray) -> tuple:
        """
        Fits multiple distributions to the data and selects
        the best one based on AICc.

        Parameters:
        ----------
        data : numpy.ndarray
            The data to which the distributions are fitted.

        Returns:
        -------
        tuple
            The best fitting distribution and weights of results
        """

        distributions = [
            scp_stats.lognorm,
            scp_stats.expon,
            scp_stats.powerlaw,
            scp_stats.norm,
            scp_stats.pareto,
        ]

        aic_values = []

        for distribution in distributions:
            current_aic = self._fit_distribution(data, distribution)
            aic_values.append(current_aic)

        weights = self._calculate_akaike_weights(aic_values)

        best_index = np.nanargmin(aic_values)
        best_distribution = distributions[best_index]

        return best_distribution, weights


    def model_choose(self, vals: pd.Series) -> tuple:
        """
        Chooses the best fitting model from a set of predefined curves
        based on AICc.

        Parameters:
        ----------
        vals : pandas.Series
            The data to which the models are fitted.

        Returns:
        -------
        tuple
            The best fit model, its name, parameters, and a DataFrame
            containing model information.
        """

        scores = {}
        parameters = {}
        expon_pred = None
        for c in [
            self.curves.linear,
            self.curves.expon,
            self.curves.expon_neg,
            self.curves.sigmoid,
        ]:
            try:
                params, covar = curve_fit(c, vals.index.values, vals.values)
                num_params = len(params)

                y_pred = c(vals.index, *params)  # type: ignore
                if c.__name__ == "expon":
                    expon_pred = y_pred
                residuals = vals.values - y_pred

                aic = len(vals.index) * np.log(np.mean(residuals**2)) + 2 * len(params)
                aicc = aic + (2 * num_params * (num_params + 1)) / (
                    len(vals) - num_params - 1
                )
                scores[c] = aicc
                parameters[c] = params

            except ValueError as e:
                logging.error(e)
                continue

        min_aicc = min(scores, key=scores.get)  # type: ignore
        min_aicc = scores[min_aicc]
        waicc = {k: v - min_aicc for k, v in scores.items()}
        waicc = {k: np.exp(-0.5 * v) for k, v in waicc.items()}
        waicc = {k: v / np.sum(list(waicc.values())) for k, v in waicc.items()}

        global_params = pd.DataFrame(
            columns=[
                "curve", "weight", "param1", "param2"
                ]
            )
        for (key1, value1), (key2, value2) in zip(waicc.items(), parameters.items()):
            if key1 == key2:
                global_params = global_params._append(
                    {
                        "curve": key2.__name__,
                        "weight": round(value1, 25),
                        "param1": round(value2[0], 25),
                        "param2": round(value2[1], 25),
                    },
                    ignore_index=True,
                )  # type: ignore

        best_fit = max(waicc, key=waicc.get)  # type: ignore
        params, _ = curve_fit(best_fit, vals.index.values, vals.values)

        return (
            best_fit(vals.index, *params),
            best_fit.__name__,
            params,
            global_params,
            expon_pred,
        )


class Stats:
    """
    A utility class for calculating various statistics and metrics
    from a TrajectoriesFrame.
    """

    @staticmethod
    def get_animals_no(data: TrajectoriesFrame) -> int:
        """
        Get the total number of unique animals in the dataset.

        Args:
            data (TrajectoriesFrame): Input data with animal trajectories.

        Returns:
            int: The number of unique animals.

        """

        return len(data.get_users())


    @staticmethod
    def get_period(data: TrajectoriesFrame) -> pd.Timedelta:
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
        data: TrajectoriesFrame
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

        unique_label_counts = data.groupby("user_id")["labels"].nunique()
        min_unique_label_count = unique_label_counts.min()

        return unique_label_counts[
            unique_label_counts == min_unique_label_count
        ]


    @staticmethod
    def get_mean_labels_no_after_filtration(data: TrajectoriesFrame) -> int:
        """
        Get the users mean number of unique labels
        after filtration.

        Args:
            data (TrajectoriesFrame): Input data with 'user_id'
                and 'labels' columns.

        Returns:
            int: Users mean no. unique label count.
        """

        unique_label_counts = data.groupby("user_id")["labels"].nunique()
        mean_unique_label_count = int(unique_label_counts.mean())

        return mean_unique_label_count


    @staticmethod
    def get_std_labels_no_after_filtration(data: TrajectoriesFrame) -> int:
        """
        Get the users std of unique labels
        after filtration.

        Args:
            data (TrajectoriesFrame): Input data with 'user_id'
                and 'labels' columns.

        Returns:
            int: Users mean no. unique label count.
        """

        unique_label_counts = data.groupby("user_id")["labels"].nunique()
        try:
            return int(unique_label_counts.std())  # type: ignore
        except:
            return 0  # type: ignore


    @staticmethod
    def get_min_records_no_before_filtration(
        data: TrajectoriesFrame
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

        records_counts = data.reset_index().groupby("user_id").datetime.count()
        min_label_count = records_counts.min()

        return records_counts[records_counts == min_label_count]


    @staticmethod
    def get_mean_records_no_before_filtration(data: TrajectoriesFrame) -> int:
        """
        Get the mean number of records
        before filtration.

        Args:
            data (TrajectoriesFrame): Input data with 'animal_id'
                and 'time' columns.

        Returns:
            int: mean number of records.
        """

        records_counts = data.reset_index().groupby("user_id").datetime.count()
        mean_label_count = int(records_counts.mean())

        return mean_label_count


    @staticmethod
    def get_std_records_no_before_filtration(data: TrajectoriesFrame) -> int:
        """
        Get the mean number of records
        before filtration.

        Args:
            data (TrajectoriesFrame): Input data with 'animal_id'
                and 'time' columns.

        Returns:
            int: mean number of records.
        """

        records_counts = data.reset_index().groupby("user_id").datetime.count()

        try:
            return int(records_counts.std())  # type: ignore
        except:
            return 0  # type: ignore


    @staticmethod
    def get_mean_periods(data: TrajectoriesFrame) -> pd.Timedelta:
        """
        Get the mean period between start and end times.

        Args:
            data (TrajectoriesFrame): Input data with 'start'
                and 'end' columns.

        Returns:
            float: The mean period in days.
        """

        return (
            data.groupby("user_id")["end"].max() - data.groupby("user_id")["start"].min()
        ).mean()  # type: ignore


    @staticmethod
    def get_min_periods(data: TrajectoriesFrame) -> pd.Timedelta:
        """
        Get the minimum period between start and end times.

        Args:
            data (TrajectoriesFrame): Input data with 'start'
                and 'end' columns.

        Returns:
            float: The minimum period in days.
        """

        return (
            data.groupby("user_id")["end"].max() - data.groupby("user_id")["start"].min()
        ).min()  # type: ignore


    @staticmethod
    def get_max_periods(data: TrajectoriesFrame) -> pd.Timedelta:
        """
        Get the maximum period between start and end times.

        Args:
            data (TrajectoriesFrame): Input data with 'start'
                and 'end' columns.

        Returns:
            float: The maximum period in days.
        """

        return (
            data.groupby("user_id")["end"].max() - data.groupby("user_id")["start"].min()
        ).max()  # type: ignore


    @staticmethod
    def get_std_periods(data: TrajectoriesFrame) -> pd.Timedelta:
        """
        Get the std of period between start and end times.

        Args:
            data (TrajectoriesFrame): Input data with 'start'
                and 'end' columns.

        Returns:
            float: The std period in days.
        """

        try:
            return (
                data.groupby("user_id")["end"].max() - data.groupby("user_id")["start"].min()
            ).std()  # type: ignore
        except:
            return 0  # type: ignore


    @staticmethod
    def get_overall_area(data: TrajectoriesFrame) -> float:
        """
        Get the overall area covered by the trajectories.

        Args:
            data (TrajectoriesFrame): Input spatial data.

        Returns:
            float: The overall area in hectares.

        """

        convex_hull = data.unary_union.convex_hull

        return round(convex_hull.area / 10000, 0)


    @staticmethod
    def get_mean_area(data: TrajectoriesFrame) -> float:
        """
        Get the mean area covered by trajectories per user.

        Args:
            data (TrajectoriesFrame): Input data.

        Returns:
            float: The mean area in hectares.
        """

        grouped = data.copy().groupby("user_id")
        areas = []
        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            areas.append(convex_hull.area / 10000)

        return round(sum(areas) / len(areas), 0)


    @staticmethod
    def get_min_area(data: TrajectoriesFrame) -> float:
        """
        Get the min area covered by trajectories per user.

        Args:
            data (TrajectoriesFrame): Input data.

        Returns:
            float: The min area in hectares.
        """

        grouped = data.copy().groupby("user_id")
        areas = []
        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            areas.append(convex_hull.area / 10000)

        return round(min(areas), 0)


    @staticmethod
    def get_max_area(data: TrajectoriesFrame) -> float:
        """
        Get the max area covered by trajectories per user.

        Args:
            data (TrajectoriesFrame): Input data.

        Returns:
            float: The max area in hectares.
        """

        grouped = data.copy().groupby("user_id")
        areas = []
        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            areas.append(convex_hull.area / 10000)

        return round(max(areas), 0)


    @staticmethod
    def get_std_area(data: TrajectoriesFrame) -> float:
        """
        Get the std of area covered by trajectories per user.

        Args:
            data (TrajectoriesFrame): Input data.

        Returns:
            float: The std of area in hectares.
        """

        grouped = data.copy().groupby("user_id")
        areas = []
        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            areas.append(convex_hull.area / 10000)

        try:
            return round(std(areas), 0)
        except:
            return 0



class DataSetStats:
    """
    A class for managing dataset statistics, storing various statistical
    measures, and saving them into a structured DataFrame.

    Attributes:
        output_dir (str): Directory to save the dataset statistics.
        record (dict): A dictionary to store temporary statistics
            before adding them to the dataset.
        stats_set (pd.DataFrame): A DataFrame containing all the
            statistics collected for different datasets.
    """

    def __init__(self, output_dir) -> None:
        """
        Initializes the DataSetStats class.

        Args:
            output_dir (str): The directory where output files
                will be stored.
        """

        self.output_dir = output_dir
        self.record = {}
        self.stats_set = pd.DataFrame(
            columns=[
                "animal",
                "animal_no",
                "animal_after_filtration",
                "time_period",
                "min_label_no",
                "mean_label_no",
                "std_label_no",
                "min_records",
                "mean_records",
                "std_records",
                "avg_duration",
                "std_duration",
                "min_duration",
                "max_duration",
                "overall_set_area",
                "average_set_area",
                "min_area",
                "max_area",
                "std_area",
                "visitation_frequency",
                "visitation_frequency_params",
                "distinct_locations_over_time",
                "distinct_locations_over_time_params",
                "jump_lengths_distribution",
                "jump_lengths_distribution_params",
                "waiting_times",
                "waiting_times_params",
                "msd_curve",
                "msd_curve_params",
                # "travel_times",
                # "travel_times_params",
                # "rog",
                # "rog_params",
                "rog_over_time",
                "rog_over_time_params",
                "msd_distribution",
                "msd_distribution_params",
                # "return_time_distribution",
                # "return_time_distribution_params",
                # "exploration_time",
                # "exploration_time_params",
                "rho",
                "gamma",
            ]
        )


    def add_data(self, data: dict) -> None:
        """
        Updates the record dictionary with new data.

        Args:
            data (dict): A dictionary containing key-value
                pairs of statistics to be added.
        """

        self.record.update(data)


    def add_record(self) -> None:
        """
        Adds the current record to the stats_set DataFrame
        and resets the record dictionary.
        """

        self.stats_set = pd.concat(
            [
                self.stats_set,
                pd.DataFrame([self.record])
            ],
            ignore_index=True)  # type: ignore
        self.record = {}



class Prepocessing:
    """
    A class containing static methods for preprocessing animal
    trajectory data.
    """

    def __init__(self) -> None:
        """
        Initializes the Prepocessing class.
        """
        pass


    @staticmethod
    def get_mean_points(data: TrajectoriesFrame) -> TrajectoriesFrame:
        """
        Computes the mean latitude and longitude for each label.

        Args:
            data (TrajectoriesFrame): The trajectory dataset.

        Returns:
            TrajectoriesFrame: A new dataset with averaged locations
                per label.
        """

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
                )  # type: ignore

        return TrajectoriesFrame(
            geometry_df.sort_values("datetime").drop_duplicates()
            )


    @staticmethod
    def set_start_stop_time(data: TrajectoriesFrame) -> TrajectoriesFrame:
        """
        Sets the start and stop time for each trajectory point.

        Args:
            data (TrajectoriesFrame): The input trajectory dataset.

        Returns:
            TrajectoriesFrame: A dataset with added start and stop times.
        """

        compressed = pd.DataFrame(
            start_end(data).reset_index()[
                [
                    "user_id",
                    "datetime",
                    "labels", ""
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
        data: TrajectoriesFrame,
        base_csr: int = const.ELLIPSOIDAL_CRS,
        target_crs: int = const.CARTESIAN_CRS,
    ) -> TrajectoriesFrame:
        """
        Converts the coordinate reference system (CRS) of the dataset.

        Args:
            data (TrajectoriesFrame): The input dataset.
            base_csr (int): The current CRS of the dataset.
            target_crs (int): The target CRS to convert to.

        Returns:
            TrajectoriesFrame: The dataset with the transformed CRS.
        """

        data_frame = data.copy().set_crs(base_csr)  # type: ignore

        return data_frame.to_crs(target_crs)


    @staticmethod
    def filter_by_min_number(
        data: TrajectoriesFrame, min_labels_no: int = const.MIN_LABEL_NO
    ) -> TrajectoriesFrame:
        """
        Filters data to include only individuals with a minimum
        number of labels.

        Args:
            data (TrajectoriesFrame): The input trajectory dataset.
            min_labels_no (int): The minimum number of labels required.

        Returns:
            TrajectoriesFrame: The filtered dataset.
        """

        data_without_nans = data[data.isna().any(axis=1)]
        distinct_locations = num_of_distinct_locations(data_without_nans)

        return TrajectoriesFrame(
            data_without_nans.loc[
                distinct_locations[distinct_locations > min_labels_no].index
            ]
        )


    @staticmethod
    def filter_by_quartiles(
        data: TrajectoriesFrame, quartile: float = const.QUARTILE
    ) -> TrajectoriesFrame:
        """
        Filters data based on quartile values of distinct locations.

        Args:
            data (TrajectoriesFrame): The input dataset.
            quartile (float): The quartile value (must be 0.25, 0.5, or 0.75).

        Returns:
            TrajectoriesFrame: The filtered dataset.

        Raises:
            ValueError: If the provided quartile value is not one of
                the allowed values.
        """

        allowed_quartiles = {0.25, 0.5, 0.75}
        if quartile not in allowed_quartiles:
            raise ValueError(
                f"Invalid quartile value: {quartile}. "
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
                    distinct_locations[distinct_locations > quartile_value].index
                ]
            )


    @staticmethod
    def filing_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes trajectory data to fill missing values by selecting
        the most visited location.

        Args:
            data (pd.DataFrame): The input trajectory dataset.

        Returns:
            pd.DataFrame: A processed dataset with missing values filled.
        """

        def longest_visited_row(groupa):
            """
            Returns the row where the individual spent the longest
            time in an interval.
            """

            if groupa.empty:
                return pd.Series(dtype=object)

            max_label = (
                groupa.groupby("labels")["duration"].sum().idxmax()
            )
            row = groupa[groupa["labels"] == max_label].iloc[0]

            return row.T


        to_conca = {}
        for uid, group in data.groupby(level=0):
            group = group[
                ~group["datetime"].duplicated()
            ]
            if len(group.labels.unique()) < 2:
                continue
            group.set_index(
                "datetime", inplace=True
            )
            group["duration"] = (
                group.index.to_series().shift(-1) - group.index
            ).dt.total_seconds()
            group["next"] = group.labels != group.labels.shift()
            group["moved"] = group.next.cumsum()
            (group.groupby("moved").duration.sum() / 3600).describe()
            group['duration'] = group['duration'].fillna(3600)

            group_resampled = group.resample("1h").apply(longest_visited_row).unstack()
            if group_resampled.index.nlevels > 1:
                group_resampled = group.resample("1h").apply(longest_visited_row)
            group_resampled = group_resampled.resample("1h").first()
            group_resampled = group_resampled.ffill().bfill()

            to_conca[uid] = group_resampled

        df = pd.DataFrame(pd.concat(to_conca))
        df["is_new"] = df.groupby(level=0, group_keys=False).apply(
            lambda x: ~x.labels.duplicated(keep="first")
        )
        df["new_sum"] = (
            df.groupby(level=0).apply(lambda x: x.is_new.cumsum()).droplevel(1)
        )

        return df



class Flexation:
    """
    A class used for identifying and fitting statistical distributions to
    segmented data based on flexation points.

    Flexation points are identified using the PELT (Pruned Exact Linear Time)
    change point detection algorithm. Once flexation points are detected,
    the data is split into left and right segments, and statistical models
    are fitted to each segment. The goodness of fit is evaluated using
    the Wasserstein distance between the empirical density and the fitted
    model.

    Attributes:
    -----------
    None
    """

    def __init__(self) -> None:
        pass


    def _calculate_penalty(self, data:ndarray) -> float:
        """
        Calculates a penalty value based on the dataset size
        and predefined sensitivity settings.

        The penalty is determined using logarithmic scaling and
        depends on the sensitivity level set
        in `const.FLEXATION_POINTS_SENSITIVITY`.

        Parameters:
        -----------
        data : ndarray
            The numerical dataset for which the penalty is calculated.

        Returns:
        --------
        float
            The computed penalty value based on the selected
                sensitivity level.
        """
        if const.FLEXATION_POINTS_SENSITIVITY == "Low":
            return 6 * log(len(data))
        elif const.FLEXATION_POINTS_SENSITIVITY == "Medium":
            return 3 * log(len(data))
        elif const.FLEXATION_POINTS_SENSITIVITY == "High":
            return 1.5 * log(len(data))


    def _calc_main_model_wasser(
        self, model_obj: distfit, data: ndarray, flexation_point: int
    ) -> float:
        """
        Calculates the Wasserstein distance between the empirical density
        and the fitted model for data split at a given flexation point.

        Parameters:
        -----------
        model_obj : distfit
            A fitted distfit model object used for density estimation.
        data : ndarray
            The numerical dataset to be analyzed.
        flexation_point : int
            The point at which the data is split into left and right subsets.

        Returns:
        --------
        float
            The Wasserstein distance between the empirical and model densities.
        """

        left_set = data[data <= flexation_point]
        right_set = data[data >= flexation_point]

        main_dfit = distfit()

        left_main_bins, left_main_empiric_density = main_dfit.density(
            X=left_set, bins=len(left_set)
        )
        right_main_bins, right_main_empiric_density = main_dfit.density(
            X=right_set, bins=len(right_set)
        )

        left_main_model_density = model_obj.model["model"].pdf(left_main_bins)
        right_main_model_density = model_obj.model["model"].pdf(right_main_bins)

        empiric_density = np.concatenate(
            (left_main_empiric_density, right_main_empiric_density)
        )
        model_density = np.concatenate(
            (left_main_model_density, right_main_model_density)
        )

        return wasserstein_distance(empiric_density, model_density)


    def _fit_mixed_models(
            self,
            data: ndarray,
            flexation_points: list
        ) -> pd.DataFrame:
        """
        Fits statistical distributions to data segments split at given
        flexation points and evaluates their goodness of fit using
        the Wasserstein distance metric.

        Parameters:
        -----------
        data : ndarray
            The numerical dataset to be analyzed.
        flexation_points : list
            A list of points at which the data is split into left
                and right subsets.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing:
            - 'point': The flexation point.
            - 'left_model': Best-fitted distribution name for the left subset.
            - 'left_score': Fit score of the left model from distfit.
            - 'left_score_calc': Calculated Wasserstein distance for
                the left subset.
            - 'right_model': Best-fitted distribution name for
                the right subset.
            - 'right_score': Fit score of the right model from distfit.
            - 'right_score_calc': Calculated Wasserstein distance for
                the right subset.
            - 'overall_score': Combined Wasserstein distance score
                for the entire dataset.
        """

        fitting_results = pd.DataFrame(
            columns=[
                "point",
                "left_model",
                "left_score",
                "left_score_calc",
                "right_model",
                "right_score",
                "right_score_calc",
                "overall_score",
            ]
        )

        for point in flexation_points:
            left_set = data[data <= point]
            right_set = data[data >= point]

            if left_set.size >= 1 and right_set.size >= 3:  # FIXME:
                left_model = distfit(
                    distr=[
                        "norm",
                        "expon",
                        "pareto",
                        "dweibull",
                        "t",
                        "genextreme",
                        "gamma",
                        "lognorm",
                        "beta",
                        "uniform",
                        "loggamma",
                        "truncexpon",
                        "truncnorm",
                        "truncpareto",
                        "powerlaw",
                    ],
                    stats="wasserstein",
                )
                right_model = distfit(
                    distr=[
                        "norm",
                        "expon",
                        "pareto",
                        "dweibull",
                        "t",
                        "genextreme",
                        "gamma",
                        "lognorm",
                        "beta",
                        "uniform",
                        "loggamma",
                        "truncexpon",
                        "truncnorm",
                        "truncpareto",
                        "powerlaw",
                    ],
                    stats="wasserstein",
                )

                left_model.fit_transform(left_set)
                right_model.fit_transform(right_set)

                left_dfit = distfit()
                left_bins, left_empiric_density = left_dfit.density(
                    X=left_set, bins=len(left_set)
                )
                left_model_density = left_model.model["model"].pdf(left_bins)

                right_dfit = distfit()
                right_bins, right_empiric_density = right_dfit.density(
                    X=right_set, bins=len(right_set)
                )
                right_model_density = right_model.model["model"].pdf(right_bins)

                left_score_calc = wasserstein_distance(
                    left_empiric_density, left_model_density
                )
                right_score_calc = wasserstein_distance(
                    right_empiric_density, right_model_density
                )

                empiric_density = np.concatenate(
                    (left_empiric_density, right_empiric_density)
                )
                model_density = np.concatenate(
                    (left_model_density, right_model_density)
                )

                overall_score = wasserstein_distance(empiric_density, model_density)

                fitting_results = fitting_results._append(
                    {
                        "point": point,  # flexation point
                        "left_model": left_model.model["name"],  # left model name
                        "left_score": left_model.summary.sort_values("score")[
                            "score"
                        ].iloc[
                            0
                        ],  # left model score from distfit
                        "left_score_calc": left_score_calc,  # left model score from our calc
                        "right_model": right_model.model["name"],  # right model name
                        "right_score": right_model.summary.sort_values("score")[  # type: ignore
                            "score"
                        ].iloc[
                            0
                        ],  # right model score from distfit
                        "right_score_calc": right_score_calc,  # right model score from our calc
                        "overall_score": overall_score,  # wasserstein calculated for mix-distribution approach
                    },
                    ignore_index=True,
                )
            else:
                pass

        return fitting_results.sort_values("right_score", ascending=True)


    def _find_flexation_points(self, data: ndarray) -> list:
        """
        Identifies flexation points in the given dataset using the
        PELT (Pruned Exact Linear Time) change point detection algorithm.

        The function applies the Pelt model with an "rbf" cost
        function to detect significant changes in the data distribution.
        A penalty value is used to control the sensitivity of change detection,
        which is calculated using `_calculate_penalty`.

        Parameters:
        -----------
        data : ndarray
            The numerical dataset in which flexation points (change points)
                are to be found.

        Returns:
        --------
        list
            A list of detected flexation points, represented as values
                from the dataset.
        """

        penalty = self._calculate_penalty(data)
        model = rpt.Pelt(model="rbf").fit(data)
        break_points_indx = model.predict(pen=penalty)

        return [data[i - 1] for i in break_points_indx]


    def find_distributions(self, model: distfit, data: ndarray):
        """
        Identifies the best-fitting distributions for segmented parts of the
        dataset based on flexation points.

        The function detects flexation points in the data, evaluates fitting
        scores, and compares the segmented model's performance to the main
        model. If a segmented approach improves the fit, it returns the left
        and right distributions along with their respective datasets.

        Parameters:
        -----------
        model : distfit
            A pre-fitted `distfit` model used as a reference for comparison.
        data : ndarray
            The numerical dataset for which distributions
                are to be identified.

        Returns:
        --------
        tuple or None
            - If segmentation improves the model fit, returns
                a tuple containing:
                - left_model (distfit): The best-fitting distribution
                    for the left segment.
                - right_model (distfit): The best-fitting distribution
                    for the right segment.
                - left_set (ndarray): The left segment of the data.
                - right_set (ndarray): The right segment of the data.
                - best_point (float): The chosen flexation point that
                    defines segmentation.
            - If no improvement is found, returns `None`.
        """

        flexation_points = self._find_flexation_points(data)
        fitting_results = self._fit_mixed_models(data, flexation_points)
        if fitting_results.shape[0] == 0:
            return None
        else:
            fitting_score = fitting_results["right_score"].iloc[0]
            best_point = fitting_results.iloc[0]["point"]

            main_model_score = self._calc_main_model_wasser(model, data, best_point)

            if main_model_score - fitting_score < 0:
                return None
            if main_model_score - fitting_score > 0:
                left_set = data[data <= best_point]
                right_set = data[data >= best_point]

                left_model = distfit(
                    distr=[
                        "norm",
                        "expon",
                        "pareto",
                        "dweibull",
                        "t",
                        "genextreme",
                        "gamma",
                        "lognorm",
                        "beta",
                        "uniform",
                        "loggamma",
                        "truncexpon",
                        "truncnorm",
                        "truncpareto",
                        "powerlaw",
                    ],
                    stats="wasserstein",
                )
                right_model = distfit(
                    distr=[
                        "norm",
                        "expon",
                        "pareto",
                        "dweibull",
                        "t",
                        "genextreme",
                        "gamma",
                        "lognorm",
                        "beta",
                        "uniform",
                        "loggamma",
                        "truncexpon",
                        "truncnorm",
                        "truncpareto",
                        "powerlaw",
                    ],
                    stats="wasserstein",
                )

                left_model.fit_transform(left_set)
                right_model.fit_transform(right_set)

                return left_model, right_model, left_set, right_set, best_point



class Laws:
    """
    A class for handling statistical data analysis and generating
    reports in PDF format.

    This class takes in a dataset, performs statistical analysis,
    applies curve fitting techniques, and generates a PDF report
    summarizing the results.

    Parameters:
    -----------
    pdf_object : FPDF
        An instance of the `FPDF` class used for generating the
            PDF report.
    stats_frame : DataSetStats
        An instance of `DataSetStats` containing statistical
            information about the dataset.
    output_path : str
        The file path where the generated PDF report will be saved.

    Attributes:
    -----------
    pdf_object : FPDF
        The PDF object used for report generation.
    output_path : str
        The location where the final report will be stored.
    stats_frame : DataSetStats
        The dataset statistics object containing calculated metrics.
    curve_fitting : DistributionFitingTools
        A toolset for fitting distributions to the dataset.
    """

    def __init__(
            self,
            pdf_object: FPDF,
            stats_frame: DataSetStats,
            output_path: str
        ):
        """
        Parameters:
        -----------
        pdf_object : FPDF
            An instance of the `FPDF` class used for generating the
                PDF report.
        stats_frame : DataSetStats
            An instance of `DataSetStats` containing statistical
                information about the dataset.
        output_path : str
            The file path where the generated PDF report will be saved.
        """

        self.pdf_object = pdf_object
        self.output_path = output_path
        self.stats_frame = stats_frame
        self.curve_fitting = DistributionFitingTools()

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
        x_position: float = 10,
        y_position: float = None,
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

        if y_position == None:
            y_position = self.pdf_object.get_y()
        else:
            y_position = y_position

        try:
            self.pdf_object.image(
                plot_obj, x=x_position, y=y_position, w=image_width, h=image_height
            )
            self.pdf_object.set_y(y_position + image_height + 10)
            plot_obj.close()
        except Exception as e:
            raise RuntimeError(f"Failed to add plot to PDF: {e}")


    def _add_pdf_curves_table(self, data:pd.DataFrame, x_offset=10, y_offset=None):
        """
        Adds a table to the PDF summarizing fitted curve parameters.

        The table includes curve names, their weights, and parameters.

        Parameters:
        -----------
        data : pd.DataFrame
            A DataFrame containing the curve fitting results with columns:
            - "curve" (str): Name of the fitted curve.
            - "weight" (float): Weight assigned to the curve.
            - "param1" (float): First parameter of the curve.
            - "param2" (float): Second parameter of the curve.
        x_offset : int, optional
            The horizontal offset for table placement in the PDF
                (default is 10).
        y_offset : int, optional
            The vertical offset for table placement. If None,
                it uses the current Y position.
        """

        if y_offset == None:
            y_offset = self.pdf_object.get_y()
        else:
            y_offset = y_offset
        self.pdf_object.set_xy(x_offset, y_offset)

        self.pdf_object.set_font("Arial", size=7)
        col_width = 25
        self.pdf_object.set_font("Arial", style="B", size=6)
        self.pdf_object.cell(col_width, 3, "Curve", border="TB", align="C")
        self.pdf_object.cell(col_width, 3, "Weight", border="TB", align="C")
        self.pdf_object.cell(col_width, 3, "Param 1", border="TB", align="C")
        self.pdf_object.cell(col_width, 3, "Param 2", border="TB", align="C")
        self.pdf_object.ln()
        self.pdf_object.set_font("Arial", size=6)

        for index, row in data.iterrows():
            self.pdf_object.cell(col_width, 3, row["curve"], border=0, align="C")
            self.pdf_object.cell(
                col_width, 3, str(round(row["weight"], 10)), border=0, align="C"
            )
            self.pdf_object.cell(
                col_width, 3, str(round(row["param1"], 10)), border=0, align="C"
            )
            self.pdf_object.cell(
                col_width, 3, str(round(row["param2"], 10)), border=0, align="C"
            )
            self.pdf_object.ln()

        self.pdf_object.cell(col_width * 4, 0, "", border="T")
        self.pdf_object.ln(1)


    def _add_pdf_msd_split_table(self, data:pd.DataFrame, x_offset:int=10, y_offset=None):
        """
        Adds a table to the PDF displaying MSD split analysis.

        The table includes ranges of Radius of Gyration (RoG)
        and associated curve parameters.

        Parameters:
        -----------
        data : pd.DataFrame
            A DataFrame containing MSD split results with columns:
            - "RoG range [km]" (str): The range of RoG values.
            - "curve" (str): Name of the fitted curve.
            - "param1" (float): First parameter of the curve.
            - "param2" (float): Second parameter of the curve.
        x_offset : int, optional
            The horizontal offset for table placement in the
                PDF (default is 10).
        y_offset : int, optional
            The vertical offset for table placement.
                If None, it uses the current Y position.
        """

        if y_offset == None:
            y_offset = self.pdf_object.get_y()
        else:
            y_offset = y_offset
        self.pdf_object.set_xy(x_offset, y_offset)

        self.pdf_object.set_font("Arial", size=7)
        col_width = 25
        self.pdf_object.set_font("Arial", style="B", size=6)
        self.pdf_object.cell(col_width, 3, "RoG range [km]", border="TB", align="C")
        self.pdf_object.cell(col_width, 3, "Curve", border="TB", align="C")
        self.pdf_object.cell(col_width, 3, "Param 1", border="TB", align="C")
        self.pdf_object.cell(col_width, 3, "Param 2", border="TB", align="C")
        self.pdf_object.ln()
        self.pdf_object.set_font("Arial", size=6)

        for index, row in data.iterrows():
            self.pdf_object.cell(
                col_width, 3, row["RoG range [km]"], border=0, align="C"
            )
            self.pdf_object.cell(col_width, 3, row["curve"], border=0, align="C")
            self.pdf_object.cell(
                col_width, 3, str(round(row["param1"], 10)), border=0, align="C"
            )
            self.pdf_object.cell(
                col_width, 3, str(round(row["param2"], 10)), border=0, align="C"
            )
            self.pdf_object.ln()

        self.pdf_object.cell(col_width * 4, 0, "", border="T")
        self.pdf_object.ln(1)


    def _add_pdf_distribution_table(self, data):
        """
        Adds a table to the PDF displaying distribution fitting results.

        The table includes distribution names, scores, and fitted parameters.

        Parameters:
        -----------
        data : pd.DataFrame
            A DataFrame containing distribution fitting results with columns:
            - "name" (str): Name of the fitted distribution.
            - "score" (float): Score associated with the fitted distribution.
            - "params" (tuple of float): Fitted parameters for
                the distribution.
        """

        self.pdf_object.set_font("Arial", style="B", size=6)
        self.pdf_object.cell(35, 3, "Distribution", border="TB", align="C")
        self.pdf_object.cell(50, 3, "Score", border="TB", align="C")
        self.pdf_object.cell(100, 3, "Params", border="TB", align="C")
        self.pdf_object.set_font("Arial", size=6)
        self.pdf_object.ln()

        for index, row in data.iterrows():
            try:
                self.pdf_object.cell(35, 3, row["name"], border=0, align="C")
                self.pdf_object.cell(
                    50, 3, str(round(row["score"], 15)), border=0, align="C"
                )
                self.pdf_object.cell(
                    100,
                    3,
                    str(tuple(round(x, 5) for x in row["params"]))
                    .replace("(", "")
                    .replace(")", ""),
                    border=0,
                    align="C",
                )
                # self.pdf_object.cell(40, 5, str(row["params"]), border=1, align="C")
                self.pdf_object.ln()
            except:
                pass
        self.pdf_object.cell(185, 0, "", border="T")
        self.pdf_object.ln(1)


    def _plot_curve(self, func_name, plot_data, y_pred, labels, exp_y_pred=None):
        """
        Plots a fitted curve along with the original data
        and saves it as a PNG file.

        Parameters:
        -----------
        func_name : str
            The name of the function, used for saving the file.
        plot_data : pd.Series or pd.DataFrame
            The original data points to be plotted as scatter points.
        y_pred : np.ndarray
            The predicted values for the fitted curve.
        labels : list of str
            A list containing x-axis and y-axis labels.
        exp_y_pred : np.ndarray, optional
            If provided, an additional exponential curve is plotted.

        Returns:
        --------
        BytesIO
            A buffer containing the saved image.
        """

        buffer = BytesIO()

        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 4.5))

        if exp_y_pred is not None and exp_y_pred.size > 0:
            plt.plot(plot_data.index, y_pred, c="k", linestyle="--", label="Sigmoid")
            plt.plot(
                plot_data.index, exp_y_pred, c="r", linestyle="-.", label="Expon neg"
            )

        else:
            plt.plot(
                plot_data.index, y_pred, c="k", linestyle="--", label="Fitted curve"
            )

        plt.scatter(plot_data.index, plot_data, color="darkturquoise", label="Data")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(labels[0], fontsize=12)
        plt.ylabel(labels[1], fontsize=12)
        plt.legend(loc="lower left", frameon=True, fontsize=10)
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.loglog()
        plt.savefig(
            os.path.join(
                self.output_path,
                f"{func_name}.png",
            )
        )
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return buffer


    def _plot_distribution(self, model, values, measure_type: str = "Values"):
        """
        Plots a histogram of values and a fitted probability distribution model.

        Parameters:
        -----------
        model : distfit
            A fitted distribution model.
        values : np.ndarray
            The dataset values to be plotted.
        measure_type : str, optional
            The type of measure for labeling the x-axis (default is "Values").

        Returns:
        --------
        tuple (BytesIO, BytesIO)
            Buffers containing the saved histogram and fitted model plots.
        """

        buffer_plot_distribution = BytesIO()
        buffer_plot_model = BytesIO()

        measure_type = (
            measure_type.replace("_", " ").replace("distribution", "").capitalize()
        )

        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 6))
        plt.hist(values, color="darkturquoise", bins=100, density=True)
        plt.xlabel(measure_type)
        plt.ylabel("Values")
        plt.loglog()

        plt.savefig(
            os.path.join(
                self.output_path,
                f"{measure_type}_distribution.png",
            )
        )
        plt.savefig(buffer_plot_distribution, format="png")
        plt.close()
        buffer_plot_distribution.seek(0)

        plt.figure(figsize=(8, 6))
        model.plot(
            pdf_properties={"color": "black", "linewidth": 2, "linestyle": "--"},
            bar_properties=None,
            cii_properties=None,
            emp_properties={"color": "darkturquoise", "linewidth": 0, "marker": "o"},
            figsize=(8, 6),
        )
        plt.xlabel(measure_type)
        plt.loglog()

        plt.savefig(
            os.path.join(
                self.output_path,
                f"{measure_type}_model_distribution.png",
            )
        )
        plt.savefig(buffer_plot_model, format="png")
        plt.close()
        buffer_plot_model.seek(0)

        return buffer_plot_distribution, buffer_plot_model


    def _plot_double_distribution(
        self,
        left_model,
        right_model,
        left_values,
        right_values,
        flexation_point,
        measure_type: str = "Values",
    ):
        """
        Plots and saves histograms for two subsets of data
        and their fitted distribution models.

        Parameters:
        -----------
        left_model : distfit
            A fitted distribution model for the left subset.
        right_model : distfit
            A fitted distribution model for the right subset.
        left_values : np.ndarray
            The left subset of data values.
        right_values : np.ndarray
            The right subset of data values.
        flexation_point : float
            The point at which the data was split.
        measure_type : str, optional
            The type of measure for labeling the x-axis
                (default is "Values").

        Returns:
        --------
        tuple (BytesIO, BytesIO)
            Buffers containing the saved histograms and model plots.
        """

        measure_type = (
            measure_type.replace("_", " ").replace("distribution", "").capitalize()
        )

        buffer_plot_distribution = BytesIO()
        buffer_plot_model = BytesIO()

        sns.set_style("whitegrid")

        plt.figure(figsize=(8, 6))
        plt.hist(
            left_values,
            bins=50,
            density=True,
            label="Left Set",
            color="grey",
        )
        plt.hist(
            right_values,
            bins=50,
            density=True,
            label="Right Set",
            color="darkturquoise",
        )
        plt.loglog()
        plt.xlabel(measure_type)
        plt.ylabel("Values")
        plt.savefig(
            os.path.join(
                self.output_path,
                f"{measure_type}_distribution_with_flexation_point_{flexation_point}.png",
            )
        )
        plt.savefig(buffer_plot_distribution, format="png")
        plt.close()
        buffer_plot_distribution.seek(0)

        left_model.plot(
            pdf_properties={"color": "black", "linewidth": 2, "linestyle": "--"},
            bar_properties=None,
            cii_properties=None,
            emp_properties={"color": "darkturquoise", "linewidth": 0, "marker": "o"},
            figsize=(8, 6),
        )

        right_model.plot(
            pdf_properties={"color": "black", "linewidth": 2, "linestyle": "--"},
            bar_properties=None,
            cii_properties=None,
            emp_properties={"color": "darkturquoise", "linewidth": 0, "marker": "o"},
            figsize=(8, 6),
        )

        plt.legend()
        plt.grid()
        plt.loglog()
        plt.savefig(
            os.path.join(
                self.output_path,
                f"{measure_type}_model_distribution_with_flexation_point_{flexation_point}.png",
            )
        )
        plt.savefig(buffer_plot_model, format="png")
        plt.close()
        buffer_plot_model.seek(0)

        return buffer_plot_distribution, buffer_plot_model


    def _plot_P_new(
        self, rho_est, gamma_est, DeltaS, S_mid, intercept, slope, nrows, n_data
    ):
        """
        Plots and saves two figures:
        1. A comparison between estimated and reference probability functions.
        2. A log-log plot of ΔS vs. S with a fitted regression line.

        Parameters:
        -----------
        rho_est : float
            Estimated scaling factor for P_new.
        gamma_est : float
            Estimated exponent for P_new.
        DeltaS : np.ndarray
            Change in S values for log-log regression.
        S_mid : np.ndarray
            Midpoints of S values for log-log regression.
        intercept : float
            Intercept of the fitted regression line in log-log plot.
        slope : float
            Slope of the fitted regression line in log-log plot.
        nrows : int
            Number of time steps for P_new estimation.
        n_data : int
            Number of data points in the dataset.

        Returns:
        --------
        tuple (BytesIO, BytesIO)
            Buffers containing the saved plots.
        """

        buffer1 = BytesIO()
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 4.5))
        plt.plot(
            np.arange(1, nrows),
            [rho_est * x ** (-gamma_est) for x in range(1, nrows)],
            label="Estimated",
            color="darkturquoise",
        )
        plt.plot(
            np.arange(1, nrows),
            [0.6 * x ** (-0.21) for x in range(1, nrows)],
            label="Paper",
            color="black",
        )
        plt.legend()
        plt.xlabel("Time steps (n)")
        plt.ylabel("P_new (estimated vs. reference)")
        plt.title("Comparison of Estimated vs. Paper Model")
        plt.savefig(
            os.path.join(
                self.output_path,
                "P new comparison of Estimated vs. Paper Model.png",
            )
        )
        plt.savefig(buffer1, format="png")
        plt.close()
        buffer1.seek(0)

        buffer2 = BytesIO()
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 4.5))
        plt.plot(
            np.log(S_mid),
            np.log(DeltaS),
            "o",
            label="Data (log-log)",
            color="darkturquoise",
        )
        X_line = np.linspace(np.log(min(S_mid)), np.log(max(S_mid)), 100)
        y_line = intercept + slope * X_line
        plt.plot(X_line, y_line, "--", label="Fit", color="black")
        plt.xlabel("ln(S)")
        plt.ylabel("ln(ΔS)")
        plt.title("log(ΔS) vs. log(S)")
        plt.legend()
        plt.savefig(
            os.path.join(
                self.output_path,
                "P_loglog.png",
            )
        )
        plt.savefig(buffer2, format="png")
        plt.close()
        buffer2.seek(0)

        return buffer1, buffer2


    def log_curve_fitting_resluts(func):
        """
        A decorator that logs and saves curve fitting results
        to a PDF report.

        The decorated function is expected to return:
        - func_name (str): Name of the function.
        - best_fit (str): Name of the best-fitting curve.
        - param_frame (pd.DataFrame): DataFrame containing
            curve parameters.
        - plot_obj (BytesIO): The plot image object.

        The results are logged in `self.stats_frame`
            and added to a PDF report.

        Parameters:
        -----------
        func : function
            The function performing curve fitting.

        Returns:
        --------
        function
            Wrapped function with additional logging and PDF output.
        """

        def wrapper(self, *args, **kwargs):
            func_name, best_fit, param_frame, plot_obj = func(self, *args, **kwargs)
            filtered_df = param_frame[param_frame["curve"] == best_fit]

            self.stats_frame.add_data({func_name: best_fit})
            self.stats_frame.add_data(
                {
                    f"{func_name}_params": filtered_df[
                        ["param1", "param2"]
                    ].values.tolist()
                }
            )

            func_name = func_name.replace("_", " ").split(" ")
            func_name[0] = func_name[0].capitalize()
            self.pdf_object.set_font("Arial", "B", size=8)
            self._add_pdf_cell(f"{' '.join(func_name)}")
            self.pdf_object.set_font("Arial", size=7)
            self._add_pdf_cell(
                f'Best fit: {best_fit} with Param 1: {filtered_df["param1"].values[0]},'
                f'Param 2: {filtered_df["param2"].values[0]}'
            )

            y_position_global = float(self.pdf_object.get_y())
            self._add_pdf_curves_table(
                param_frame, x_offset=10, y_offset=y_position_global + 13
            )
            self._add_pdf_plot(
                plot_obj=plot_obj,
                image_width=80,
                image_height=45,
                x_position=125,
                y_position=y_position_global,
            )

        return wrapper


    def log_distribution_fitting_resluts(func):
        """
        A decorator that logs and saves distribution fitting
        results to a PDF report.

        The decorated function is expected to return:
        - results[0] (str): The name of the fitted distribution.
        - results[1] (distfit object): Fitted distribution model.
        - results[2] (BytesIO): Distribution plot.
        - results[3] (BytesIO): Model plot.
        - results[4] (tuple, optional): If present, contains
            information about a flexion point.

        If a flexion point is detected, it logs both left
        and right distributions and plots them separately.

        Parameters:
        -----------
        func : function
            The function performing distribution fitting.

        Returns:
        --------
        function
            Wrapped function with additional logging and PDF output.
        """

        def wrapper(self, *args, **kwargs):
            results = func(self, *args, **kwargs)

            self.stats_frame.add_data({results[0]: results[1].model["name"]})
            self.stats_frame.add_data(
                {f"{results[0]}_params": results[1].model["params"]}
            )

            func_name = results[0].replace("_", " ").split(" ")
            func_name[0] = func_name[0].capitalize()
            self.pdf_object.set_font("Arial", "B", size=8)
            self._add_pdf_cell(f"{' '.join(func_name)}")
            self.pdf_object.set_font("Arial", size=7)
            self._add_pdf_cell(
                f'Best fit: {results[1].model["name"]} with params: {results[1].model["params"]}'
            )

            self._add_pdf_distribution_table(
                results[1].summary[["name", "score", "params"]]
            )
            y_position_global = float(self.pdf_object.get_y())
            self._add_pdf_plot(
                results[2], 80, 60, x_position=10, y_position=y_position_global
            )
            self._add_pdf_plot(
                results[3], 80, 60, x_position=110, y_position=y_position_global
            )

            if len(results) == 5:
                self._add_pdf_cell(
                    f"At point {results[4][4]}, the flexion point of the distributions was found"
                )
                self._add_pdf_cell(
                    f'Left distribution is {results[4][0].model["name"]} with params: {results[4][0].model["params"]}'
                )
                self._add_pdf_cell(
                    f'Right distribution is {results[4][1].model["name"]} with params: {results[4][1].model["params"]}'
                )
                self._add_pdf_distribution_table(
                    results[4][1].summary[["name", "score", "params"]]
                )
                plot_distribution, plot_models = self._plot_double_distribution(
                    results[4][0],
                    results[4][1],
                    results[4][2],
                    results[4][3],
                    results[4][4],
                    results[0],
                )
                y_position_global = float(self.pdf_object.get_y())
                self._add_pdf_plot(
                    plot_distribution,
                    80,
                    60,
                    x_position=10,
                    y_position=y_position_global,
                )
                self._add_pdf_plot(
                    plot_models, 80, 60, x_position=110, y_position=y_position_global
                )
            self.pdf_object.add_page()

        return wrapper


    def log_pnew_estimation(func):
        """
        A decorator that logs and saves P_new estimation results
        to a PDF report.

        The decorated function is expected to return:
        - rho_est (float): Estimated rho parameter.
        - gamma_est (float): Estimated gamma parameter.
        - DeltaS (np.ndarray): Change in S values for log-log regression.
        - S_mid (np.ndarray): Midpoints of S values for log-log regression.
        - intercept (float): Intercept of the fitted regression line.
        - slope (float): Slope of the fitted regression line.
        - nrows (int): Number of time steps for estimation.
        - n_data (int): Number of data points.

        The results are logged in `self.stats_frame`
        and visualized in a PDF.

        Parameters:
        -----------
        func : function
            The function performing P_new estimation.

        Returns:
        --------
        function
            Wrapped function with additional logging and PDF output.
        """

        def wrapper(self, *args, **kwargs):
            rho_est, gamma_est, DeltaS, S_mid, intercept, slope, nrows, n_data = func(
                self, *args, **kwargs
            )  # type: ignore

            self.stats_frame.add_data({"rho": rho_est})
            self.stats_frame.add_data({"gamma": gamma_est})

            self.pdf_object.set_font("Arial", "B", size=8)
            self._add_pdf_cell("Pnew estimation")

            self.pdf_object.ln()
            self.pdf_object.ln()
            self.pdf_object.set_font("Arial", size=7)
            self._add_pdf_cell(f"gamma  = {gamma_est:.4f}")
            self._add_pdf_cell(f"rho  = {rho_est:.4f}")
            y_position_global = float(self.pdf_object.get_y())
            plot_obj1, plot_obj2 = self._plot_P_new(
                rho_est, gamma_est, DeltaS, S_mid, intercept, slope, nrows, n_data
            )
            self._add_pdf_plot(
                plot_obj=plot_obj1,
                image_width=80,
                image_height=45,
                x_position=10,
                y_position=y_position_global,
            )
            self._add_pdf_plot(
                plot_obj=plot_obj2,
                image_width=80,
                image_height=45,
                x_position=110,
                y_position=y_position_global,
            )

        return wrapper


    def log_msd_split(func):
        """
        A decorator that logs and saves MSD split results
        to a PDF report.

        The decorated function is expected to return:
        - msd_results (pd.DataFrame): MSD split results
            including range and curve parameters.
        - plot_obj (BytesIO): The plot image object.

        The results are stored in `self.stats_frame`
        and included in a PDF report.

        Parameters:
        -----------
        func : function
            The function performing MSD split analysis.

        Returns:
        --------
        function
            Wrapped function with additional logging and PDF output.
        """

        def wrapper(self, *args, **kwargs):
            msd_results, plot_obj = func(self, *args, **kwargs)  # type: ignore
            msd_curve = "; ".join(
                msd_results["RoG range [km]"] + ": " + msd_results["curve"]
            )
            msd_curve_params = "; ".join(
                msd_results["RoG range [km]"]
                + "("
                + msd_results["curve"]
                + ")"
                + ":"
                + msd_results["param1"].astype(str)
                + ","
                + msd_results["param2"].astype(str)
            )

            self.stats_frame.add_data({"msd_curve": msd_curve})
            self.stats_frame.add_data({"msd_curve_params": msd_curve_params})

            self.pdf_object.set_font("Arial", "B", size=8)
            self._add_pdf_cell("MSD split")
            y_position_global = float(self.pdf_object.get_y())
            self.pdf_object.ln(4)
            self._add_pdf_msd_split_table(msd_results)
            self._add_pdf_plot(
                plot_obj=plot_obj,
                image_width=80,
                image_height=45,
                x_position=125,
                y_position=y_position_global,
            )

        return wrapper


    def check_curve_fit(func):
        """
        A decorator that evaluates curve fitting results
        and generates a corresponding plot.

        The decorated function is expected to return:
        - best_fit (str): Name of the best-fitting curve.
        - param_frame (pd.DataFrame): DataFrame containing
            curve parameters.
        - y_pred (np.ndarray): Predicted values based on
            the curve fitting.
        - exp_y_pred (np.ndarray or None): Exponential
            curve prediction (if applicable).
        - plot_data (pd.Series or np.ndarray): Data used for fitting.
        - labels (list): Axis labels for the plot.

        If `best_fit` is not among {"linear", "expon", "expon_neg"},
        both the fitted curve and an additional exponential curve
        are plotted. Otherwise, only the fitted curve is plotted.

        Parameters:
        -----------
        func : function
            The function performing curve fitting.

        Returns:
        --------
        function
            Wrapped function that returns:
            - func.__name__ (str): Name of the function.
            - best_fit (str): Best fitting curve name.
            - param_frame (pd.DataFrame): Parameters of
                the best fit.
            - plot_obj (BytesIO): Image object containing
                the generated plot.
        """

        def wrapper(self, *args, **kwargs):
            best_fit, param_frame, y_pred, exp_y_pred, plot_data, labels = func(
                self, *args, **kwargs
            )  # type: ignore
            if best_fit not in {"linear", "expon", "expon_neg"}:
                plot_obj = self._plot_curve(
                    func.__name__, plot_data, y_pred, labels, exp_y_pred
                )
                return func.__name__, best_fit, param_frame, plot_obj
            else:
                plot_obj = self._plot_curve(func.__name__, plot_data, y_pred, labels)
                return func.__name__, best_fit, param_frame, plot_obj

        return wrapper


    def check_distribution_fit(func):
        """
        A decorator that evaluates the results of distribution
        fitting and detects flexation points.

        The decorated function is expected to return:
        - model (distfit object): The best-fitted distribution
            model.
        - data (pd.Series or np.ndarray): Data used for
            distribution fitting.

        The function generates:
        - A histogram plot of the fitted distribution.
        - A model plot displaying the probability density function (PDF).

        If the fitted distribution is not in `const.DISTRIBUTIONS`
        and the data has at least 4 observations, or if specific conditions
        apply (e.g., Pareto or Lognormal with parameter > 2),
        an attempt is made to detect flexation points in the distribution.

        Parameters:
        -----------
        func : function
            The function performing distribution fitting.

        Returns:
        --------
        function
            Wrapped function that returns:
            - func.__name__ (str): Name of the function.
            - model (distfit object): Best-fitting distribution model.
            - plot_distribution_obj (BytesIO): Image object containing
                the histogram plot.
            - plot_model_obj (BytesIO): Image object containing the PDF plot.
            - flexation_point_detection_results (tuple, optional):
            If flexation points are found, returns details of left
            and right distributions.
        """

        def wrapper(self, *args, **kwargs):
            model, data = func(self, *args, **kwargs)

            plot_distribution_obj, plot_model_obj = self._plot_distribution(
                model, data, func.__name__
            )
            if (
                model.model["name"] not in const.DISTRIBUTIONS and data.shape[0] >= 4
            ):  # FIXME:
                flexation_point_detection_results = Flexation().find_distributions(
                    model, np.sort(data.values)
                )
                if flexation_point_detection_results != None:

                    return (
                        func.__name__,
                        model,
                        plot_distribution_obj,
                        plot_model_obj,
                        flexation_point_detection_results,
                    )
                else:
                    return func.__name__, model, plot_distribution_obj, plot_model_obj
            elif model.model["name"] == "pareto" and model.model["params"][0] > 2:
                flexation_point_detection_results = Flexation().find_distributions(
                    model, np.sort(data.values)
                )
                if flexation_point_detection_results != None:

                    return (
                        func.__name__,
                        model,
                        plot_distribution_obj,
                        plot_model_obj,
                        flexation_point_detection_results,
                    )
                else:
                    return func.__name__, model, plot_distribution_obj, plot_model_obj
            elif model.model["name"] == "lognorm" and model.model["params"][0] > 2:
                flexation_point_detection_results = Flexation().find_distributions(
                    model, np.sort(data.values)
                )
                if flexation_point_detection_results != None:

                    return (
                        func.__name__,
                        model,
                        plot_distribution_obj,
                        plot_model_obj,
                        flexation_point_detection_results,
                    )
                else:
                    return func.__name__, model, plot_distribution_obj, plot_model_obj
            else:
                return func.__name__, model, plot_distribution_obj, plot_model_obj

        return wrapper


    @log_curve_fitting_resluts
    @check_curve_fit
    def visitation_frequency(
        self, data: TrajectoriesFrame, min_labels_no: int
    ) -> tuple:
        """
        Computes the visitation frequency of locations based
        on trajectory data.

        The function:
        - Computes visitation frequency from trajectory data.
        - Averages visitation frequency over a specified
            minimum number of labels.
        - Performs curve fitting to find the best function
            describing the relationship.
        - Returns fitting results, including predictions
            and parameter estimates.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing location visit information.
        min_labels_no : int
            Minimum number of labels required for averaging
                the visitation frequency.

        Returns:
        --------
        tuple:
            - best_fit (str): Best fitting curve name.
            - global_params (pd.DataFrame): Parameters of the best fit.
            - y_pred (np.ndarray): Predicted visitation frequencies.
            - exp_y_pred (np.ndarray or None): Exponential curve
                predictions (if applicable).
            - avg_vf (pd.Series): Averaged visitation frequency data.
            - labels (list): Axis labels for the plot.
        """

        vf = visitation_frequency(data)
        avg_vf = rowwise_average(vf, row_count=min_labels_no)
        avg_vf.index += 1
        vf.groupby(level=0).size().median()
        avg_vf = avg_vf[~avg_vf.isna()]

        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            self.curve_fitting.model_choose(avg_vf)
        )

        return best_fit, global_params, y_pred, expon_y_pred, avg_vf, ["Rank", "f"]


    @log_distribution_fitting_resluts
    @check_distribution_fit
    def jump_lengths_distribution(self, data: TrajectoriesFrame) -> tuple:
        """
        Computes the distribution of jump lengths in a trajectory dataset.

        The function:
        - Extracts jump lengths from the dataset.
        - Removes zero values and converts the data into a distribution.
        - Fits multiple statistical distributions and selects the best fit.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing movement data.

        Returns:
        --------
        tuple:
            - model (distfit object): The fitted distribution model.
            - jl (pd.Series): The processed jump length data.
        """

        jl = jump_lengths(data)
        jl = jl[jl != 0]
        jl_dist = convert_to_distribution(jl, num_of_classes=20)

        # Fit to find the best theoretical distribution
        jl = jl[~jl.isna()]
        model = distfit(
            distr=[
                "norm",
                "expon",
                "pareto",
                "dweibull",
                "t",
                "genextreme",
                "gamma",
                "lognorm",
                "beta",
                "uniform",
                "loggamma",
                "truncexpon",
                "truncnorm",
                "truncpareto",
                "powerlaw",
            ],
            stats="wasserstein",
        )
        model.fit_transform(jl.values)

        return model, jl


    @log_distribution_fitting_resluts
    @check_distribution_fit
    def waiting_times(self, data: TrajectoriesFrame) -> tuple:
        """
        Computes the distribution of waiting times in a trajectory dataset.

        The function:
        - Calculates the waiting times as the difference between start
            and end times.
        - Removes missing and zero values.
        - Fits multiple statistical distributions and selects the best fit.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing time-based movement data.

        Returns:
        --------
        tuple:
            - model (distfit object): The fitted distribution model.
            - wt (pd.Series): The processed waiting time data.
        """

        data_set = data.copy()
        try:
            wt = data_set.groupby(level=0).apply(
                lambda x: (x.end - x.start).dt.total_seconds().round()
            )
        except:
            wt = (data_set["end"] - data_set["start"]).dt.total_seconds().round()

        wt = wt[~wt.isna()]  # type: ignore
        wt = wt[wt != 0]

        # Fit to find the best theoretical distribution
        model = distfit(
            distr=[
                "norm",
                "expon",
                "pareto",
                "dweibull",
                "t",
                "genextreme",
                "gamma",
                "lognorm",
                "beta",
                "uniform",
                "loggamma",
                "truncexpon",
                "truncnorm",
                "truncpareto",
                "powerlaw",
            ],
            stats="wasserstein",
        )
        model.fit_transform(wt.values)

        return model, wt


    @log_distribution_fitting_resluts
    @check_distribution_fit
    def travel_times(self, data: TrajectoriesFrame) -> tuple:
        """
        Computes the distribution of travel times between locations.

        The function:
        - Calculates travel times as the time difference between
            consecutive trajectory points.
        - Removes missing values.
        - Fits multiple statistical distributions and selects the best fit.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing time-based movement data.

        Returns:
        --------
        tuple:
            - model (distfit object): The fitted distribution model.
            - tt (pd.Series): The processed travel time data.
        """

        data_set = data.copy()
        try:
            tt = (
                data_set.groupby(level=0)
                .progress_apply(lambda x: x.shift(-1).start - x.end)  # type: ignore
                .reset_index(level=[1, 2], drop=True)
            )
        except:
            shifted_start = data_set["start"].shift(-1)
            tt = shifted_start - data_set["end"]
            tt = tt.reset_index(drop=True)

        tt = tt.dt.total_seconds()
        tt = tt[~tt.isna()]

        # Fit to find the best theoretical distribution
        model = distfit(
            distr=[
                "norm",
                "expon",
                "pareto",
                "dweibull",
                "t",
                "genextreme",
                "gamma",
                "lognorm",
                "beta",
                "uniform",
                "loggamma",
                "truncexpon",
                "truncnorm",
                "truncpareto",
                "powerlaw",
            ],
            stats="wasserstein",
        )
        model.fit_transform(tt.values)

        return model, tt


    @log_distribution_fitting_resluts
    @check_distribution_fit
    def rog(self, data: TrajectoriesFrame) -> tuple:
        """
        Computes the distribution of the radius of gyration (RoG).

        The function:
        - Computes the radius of gyration for each trajectory.
        - Fits multiple statistical distributions and selects the best fit.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing spatial movement data.

        Returns:
        --------
        tuple:
            - model (distfit object): The fitted distribution model.
            - rog (pd.Series): The processed radius of gyration data.
        """

        rog = radius_of_gyration(data, time_evolution=False)

        # Fit to find the best theoretical distribution
        model = distfit(
            distr=[
                "norm",
                "expon",
                "pareto",
                "dweibull",
                "t",
                "genextreme",
                "gamma",
                "lognorm",
                "beta",
                "uniform",
                "loggamma",
                "truncexpon",
                "truncnorm",
                "truncpareto",
                "powerlaw",
            ],
            stats="wasserstein",
        )
        model.fit_transform(rog.values)

        return model, rog


    @log_curve_fitting_resluts
    @check_curve_fit
    def rog_over_time(
        self,
        data: TrajectoriesFrame,
        min_records_no: int
    ) -> tuple:
        """
        Computes the evolution of the radius of gyration (RoG) over time.

        The function:
        - Computes the radius of gyration for each trajectory over time.
        - Averages the RoG values using a specified number of records.
        - Fits a curve to model the evolution of RoG.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing movement data.
        min_records_no : int
            Minimum number of records required for averaging.

        Returns:
        --------
        tuple:
            - best_fit (str): Name of the best fitting model.
            - global_params (dict): Parameters of the best fit.
            - y_pred (ndarray): Predicted values from the curve fit.
            - expon_y_pred (ndarray): Alternative exponential model
                predictions.
            - avg_rog (pd.Series): Averaged RoG values over time.
            - ["Time", "Values"] (list): Column labels.
        """

        rog = radius_of_gyration(data, time_evolution=True)
        avg_rog = rowwise_average(rog, row_count=min_records_no)
        avg_rog = avg_rog[~avg_rog.isna()]

        # model selection
        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(avg_rog)
        )

        return (
            best_fit,
            global_params,
            y_pred,
            expon_y_pred,
            avg_rog,
            ["Time", "Values"],
        )


    @log_distribution_fitting_resluts
    @check_distribution_fit
    def msd_distribution(self, data: TrajectoriesFrame) -> tuple:
        """
        Computes the distribution of mean squared displacement (MSD).

        The function:
        - Computes MSD without time evolution.
        - Fits multiple statistical distributions and selects the best fit.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing movement data.

        Returns:
        --------
        tuple:
            - model (distfit object): The fitted distribution model.
            - msd (pd.Series): The computed MSD values.
        """

        msd = mean_square_displacement(
            data,
            time_evolution=False,
            from_center=True
        )
        model = distfit(
            distr=[
                "norm",
                "expon",
                "pareto",
                "dweibull",
                "t",
                "genextreme",
                "gamma",
                "lognorm",
                "beta",
                "uniform",
                "loggamma",
                "truncexpon",
                "truncnorm",
                "truncpareto",
                "powerlaw",
            ],
            stats="wasserstein",
        )
        model.fit_transform(msd.values)

        return model, msd


    @log_curve_fitting_resluts
    @check_curve_fit
    def msd_curve(
        self,
        data: TrajectoriesFrame,
        min_records_no: int
    ) -> tuple:
        """
        Computes the mean squared displacement (MSD) curve over time.

        The function:
        - Computes MSD over time.
        - Averages MSD values using a specified number of records.
        - Fits a curve to model MSD behavior.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing movement data.
        min_records_no : int
            Minimum number of records required for averaging.

        Returns:
        --------
        tuple:
            - best_fit (str): Name of the best fitting model.
            - global_params (dict): Parameters of the best fit.
            - y_pred (ndarray): Predicted values from the curve fit.
            - expon_y_pred (ndarray): Alternative exponential model
                predictions.
            - avg_msd (pd.Series): Averaged MSD values over time.
            - ["t", "MSD"] (list): Column labels.
        """

        msd = mean_square_displacement(
            data,
            time_evolution=True,
            from_center=False
        )
        avg_msd = rowwise_average(msd, row_count=min_records_no)
        avg_msd = avg_msd[~avg_msd.isna()]
        # model selection
        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(avg_msd)
        )

        return (
            best_fit,
            global_params,
            y_pred,
            expon_y_pred,
            avg_msd,
            ["t", "MSD"]
        )


    @log_distribution_fitting_resluts
    @check_distribution_fit
    def return_time_distribution(self, data: TrajectoriesFrame) -> tuple:
        """
        Computes the distribution of return times.

        The function:
        - Identifies return times based on revisited locations.
        - Fits a statistical distribution to the return time values.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing labeled locations and timestamps.

        Returns:
        --------
        tuple:
            - model (distfit object): The fitted distribution model.
            - rt (pd.Series): The computed return time values.
        """

        to_concat = {}
        data_set = data.copy()
        for uid, vals in tqdm(
            data_set.groupby(level=0),
            total=len(pd.unique(data_set.index.get_level_values(0))),
        ):
            vals = vals.sort_index()[["labels", "start", "end"]]

            vals["new_place"] = ~vals["labels"].duplicated(keep="first")
            vals["islands"] = vals["new_place"] * (
                (vals["new_place"] != vals["new_place"].shift(1)).cumsum()
            )
            vals["islands_reach"] = vals["islands"].shift()
            vals["islands"] = vals[["islands", "islands_reach"]].max(axis=1)

            vals = vals.drop("islands_reach", axis=1)
            vals = vals[vals.islands > 0]

            result = vals.groupby("islands").apply(
                lambda x: x.iloc[-1].start - x.iloc[0].start if len(x) > 0 else None  # type: ignore
            )
            result = result.dt.total_seconds()
            to_concat[uid] = result

        rt = pd.concat(to_concat)
        rt = rt.reset_index(level=1, drop=True)
        rt = rt[rt != 0]
        rt = pd.concat(to_concat)
        rt = rt[rt != 0]

        model = distfit(stats="wasserstein")
        model.fit_transform(rt.values)

        return model, rt


    @log_distribution_fitting_resluts
    @check_distribution_fit
    def exploration_time(self, data: TrajectoriesFrame) -> tuple:
        """
        Computes the distribution of exploration times.

        The function:
        - Identifies exploration periods as the time spent in
            newly visited locations.
        - Fits a statistical distribution to the exploration time values.

        Parameters:
        -----------
        data : TrajectoriesFrame
            A trajectory dataset containing labeled locations and timestamps.

        Returns:
        --------
        tuple:
            - model (distfit object): The fitted distribution model.
            - et (pd.Series): The computed exploration time values.
        """

        to_concat = {}
        data_set = data.copy()
        for uid, vals in tqdm(
            data_set.groupby(level=0),
            total=len(pd.unique(data_set.index.get_level_values(0))),
        ):
            vals = vals.sort_index()[["labels", "start", "end"]]

            vals["old_place"] = vals["labels"].duplicated(keep="first")
            vals["islands"] = vals["old_place"] * (
                (vals["old_place"] != vals["old_place"].shift(1)).cumsum()
            )
            vals["islands_reach"] = vals["islands"].shift()
            vals["islands"] = vals[["islands", "islands_reach"]].max(axis=1)

            vals = vals.drop("islands_reach", axis=1)
            vals = vals[vals.islands > 0]

            result = vals.groupby("islands").apply(
                lambda x: x.iloc[-1].start - x.iloc[0].start if len(x) > 0 else None  # type: ignore
            )
            if result.size == 0:
                continue
            result = result.dt.total_seconds()
            to_concat[uid] = result

        et = pd.concat(to_concat)
        et = et.reset_index(level=1, drop=True)
        et = et[et != 0]

        model = distfit(stats="wasserstein")
        model.fit_transform(et.values)

        return model, et


    @log_msd_split
    def msd_curve_split(self, data):
        """
        Computes and groups MSD curves based on the radius of gyration (RoG).

        The function:
        - Converts trajectory data to a GeoDataFrame.
        - Computes the center of mass and starting points for each trajectory.
        - Filters data to only include new explorations.
        - Computes MSD and RoG values for each group.
        - Groups MSD curves based on RoG bins and fits curves to them.

        Parameters:
        -----------
        data : pd.DataFrame
            A dataset containing longitude and latitude coordinates.

        Returns:
        --------
        tuple:
            - msd_results (pd.DataFrame): Summary of MSD curve fitting for
                different RoG ranges.
            - buffer (BytesIO): A buffer containing the saved MSD plot.
        """

        gdf = gpd.GeoDataFrame(
            data, geometry=gpd.points_from_xy(data["lon"], data["lat"])
        )
        gdf.crs = 4326
        gdf = gdf.to_crs(3857)
        com = gdf.groupby(level=0).apply(
            lambda z: Point(z.geometry.x.mean(), z.geometry.y.mean())
        )
        starting_points = (
            gdf.groupby(level=0).head(1).droplevel(1).geometry
        )
        to_concat_msd = []
        to_concat_rog = {}
        for ind, vals in gdf.groupby(level=0):
            vals["is_new"] = ~vals.labels.duplicated(keep="first")
            vals = vals[vals["is_new"]]
            vals = vals.dropna()[1:]
            msd_ind = vals.distance(starting_points.loc[ind]) ** 2
            rog_ind = vals.distance(com.loc[ind]) ** 2
            msd_ind = groupwise_expansion(msd_ind)
            to_concat_msd.append(msd_ind)
            to_concat_rog[ind] = np.sqrt(rog_ind).mean()
        final_msd = pd.concat(to_concat_msd)
        final_rog = pd.DataFrame.from_dict(
            to_concat_rog, orient="index"
        )

        final_rog["bins"] = pd.cut(
            final_rog.values.ravel(),
            bins=[
                0, 2e3, 4e3, 8e3, 16e3, 32e3, 64e3, 128e3, 256e3, 512e3, 1024e3
            ],
        )
        msd_results = pd.DataFrame(
            columns=["RoG range [km]", "curve", "param1", "param2"]
        )
        buffer = BytesIO()
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 4.5))
        for ind, vals in final_rog.groupby("bins"):
            if vals.empty:
                continue
            if vals.shape[0] < 2:
                continue
            msd_pick = final_msd.loc[vals.index.values]
            msd_rows = int(
                msd_pick.groupby(level=0).apply(lambda x: len(x)).min()
            )  # find min number of rows for
            # individuals
            msd_pick_avg = rowwise_average(msd_pick, msd_rows)
            try:  # now average rowwise
                msd_chosen = DistributionFitingTools().model_choose(
                    msd_pick_avg
                )  # pick model
            except ValueError:
                continue
            if len(msd_chosen[0]) < 3:
                continue
            avg_error = r2_score(
                np.log(msd_pick_avg[1:]), np.log(msd_chosen[0][1:])
            )
            if avg_error < 0.25:
                continue
            msd_results = msd_results._append(
                {
                    "RoG range [km]": f"{(int(ind.left/1000))}-{(int(ind.right/1000))}",
                    "curve": msd_chosen[1],
                    "param1": msd_chosen[2][0],
                    "param2": msd_chosen[2][1],  # type: ignore
                },
                ignore_index=True,
            )
            plt.scatter(np.arange(msd_rows), msd_pick_avg.values)
            plt.plot(
                np.arange(msd_pick_avg.shape[0]),
                msd_chosen[0],
                label=f"RoG:<{(int(ind.right/1000))}km ({vals.shape[0]})"
            )  # type: ignore

        plt.legend()
        plt.loglog()
        plt.xlabel("t")
        plt.ylabel("MSD")
        plt.savefig(
            os.path.join(
                self.output_path,
                "MSD split.png",
            )
        )
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        return msd_results, buffer


    @log_curve_fitting_resluts
    @check_curve_fit
    def distinct_locations_over_time(
        self,
        nrows:int,
        n_data:int,
        data:pd.Series
    ) -> tuple:
        """
        Computes the number of distinct locations visited over time.

        The function:
        - Computes the cumulative number of distinct locations visited.
        - Fits a curve to model the growth of visited locations over time.

        Parameters:
        -----------
        nrows : int
            Number of rows in the dataset.
        n_data : int
            Total data points available.
        data : pd.Series
            A dataset containing the count of distinct locations.

        Returns:
        --------
        tuple:
            - best_fit (str): Name of the best fitting model.
            - global_params (dict): Parameters of the best fit.
            - y_pred (ndarray): Predicted values from the curve fit.
            - expon_y_pred (ndarray): Alternative exponential model
                predictions.
            - pd.Series(data): The input data as a pandas Series.
            - ["t", "S(t)"] (list): Column labels.
        """

        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(pd.Series(data))
        )

        return (
            best_fit,
            global_params,
            y_pred,
            expon_y_pred,
            pd.Series(data),
            ["t", "S(t)"],
        )


    @log_pnew_estimation
    def estimate_pnew(self, filled_data:pd.DataFrame) -> tuple:
        """
        Estimates parameters (rho, gamma) for the probability of
        discovering new locations.

        The function:
        - Computes the rate of discovery of new locations (P_new).
        - Fits a linear model in log-log space to estimate the parameters.

        Parameters:
        -----------
        nrows : int
            Number of rows in the dataset.
        n_data : int
            Total data points available.
        S_t : list or np.array
            Sequence of distinct locations visited over time.

        Returns:
        --------
        tuple:
            - rho_hat (float): Estimated rho parameter.
            - gamma_hat (float): Estimated gamma parameter.
            - DeltaS (np.array): Changes in S(t).
            - S_mid (np.array): Midpoints of S(t).
            - intercept (float): Intercept of the log-log regression.
            - slope (float): Slope of the log-log regression (negative gamma).
            - nrows (int): Number of rows in the dataset.
            - n_data (int): Total data points available.
        """
        filled_data.to_csv("filled_data.csv")
        nrows = int(filled_data.groupby(level=0).apply(lambda x: len(x)).min())
        t_array = np.arange(1, nrows + 1)

        n_array = filled_data.next.groupby(level=0).apply(lambda x: x.cumsum())
        n_array = np.array([n_array.groupby(level=0).nth(x).mean() for x in range(nrows)])


        S_array = np.array([filled_data.groupby(level=0)['new_sum'].nth(x).mean() for x in range(nrows)])

        fit_n = powerlaw.Fit(n_array + 1e-10, discrete=True, verbose=False)
        beta_est = fit_n.power_law.alpha - 1
        C_est = np.exp(np.mean(np.log(n_array + 1e-10) - beta_est * np.log(t_array)))

        fit_s = powerlaw.Fit(S_array + 1e-10, discrete=True, verbose=False)
        alpha_est = fit_s.power_law.alpha - 1
        K_est = np.exp(np.mean(np.log(S_array + 1e-10) - alpha_est * np.log(t_array)))

        # Gamma from theory
        gamma_est = beta_est / alpha_est - 1

        # Estimate rho from prefactor relation
        num = np.log(K_est) - np.log(1 + gamma_est)
        den = 1.0 / (1.0 + gamma_est)
        temp = num / den - np.log(C_est)
        rho_est = np.exp(temp)

        gamma = gamma_est
        rho = rho_est


        DeltaS = S_array[1:] - S_array[:-1]
        S_mid = S_array[:-1]
        slope = -gamma
        intercept = temp

        mask = (DeltaS > 0) & (S_mid > 0)
        DeltaS = DeltaS[mask]
        S_mid = S_mid[mask]

        # S_t = np.array(S_t)
        # DeltaS = S_t[1:] - S_t[:-1]
        # S_mid = S_t[:-1]
        # mask = (DeltaS > 0) & (S_mid > 0)
        # DeltaS = DeltaS[mask]
        # S_mid = S_mid[mask]

        # X = np.log(S_mid).reshape(-1, 1)  # predictor
        # y = np.log(DeltaS)  # response

        # model = LinearRegression()
        # model.fit(X, y)

        # slope = model.coef_[0]
        # intercept = model.intercept_
        # gamma_est = -slope
        # rho_est = np.exp(intercept)

        return (
            rho,
            gamma,
            DeltaS,
            S_mid,
            intercept,
            slope,
            nrows,
            t_array
            )



class ScalingLawsCalc:
    """
    A class for processing animal trajectory data, performing
    statistical analyses, and generating a PDF report with results.

    Attributes:
        data (TrajectoriesFrame): The trajectory data of animals.
        animal_name (str): Name of the dataset.
        output_dir (str): Directory to store the output files.
        output_dir_animal (str): Path for the specific dataset output.
        pdf (FPDF): PDF object to generate reports.
        stats_frame (DataSetStats): Object for tracking dataset statistics.
    """

    def __init__(
        self,
        data: TrajectoriesFrame,
        data_name: str,
        output_dir: str,
        stats_frame: DataSetStats,
    ) -> None:
        """
        Initializes the ScalingLawsCalc class.

        Args:
            data (TrajectoriesFrame): The trajectory dataset.
            data_name (str): The name of the dataset.
            output_dir (str): Directory to save output files.
            stats_frame (DataSetStats): Object to store statistics
                of the dataset.

        Raises:
            FileExistsError: If the output directory for the dataset
                already exists.
        """

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
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_font("Arial", "B", size=12)
        self.pdf.cell(
            200, 10, text=f"{self.animal_name.replace('_',' ')}", ln=True, align="C"
        )
        self.pdf.set_font("Arial", size=9)
        self.pdf.ln(5)


    def _preprocess_data(self) -> TrajectoriesFrame:
        """
        Preprocesses the trajectory data by filtering, converting
        coordinate systems, and calculating summary statistics.

        Returns:
            TrajectoriesFrame: The preprocessed trajectory data.
        """

        preproc = Prepocessing()
        stats = Stats()
        mean_points_values = preproc.get_mean_points(self.data)
        compressed_points = preproc.set_start_stop_time(mean_points_values)
        converted_to_cartesian = preproc.set_crs(compressed_points)
        filtered_animals = preproc.filter_by_quartiles(converted_to_cartesian)
        self.stats_frame.add_data({"animal_no": stats.get_animals_no(self.data)})
        self.stats_frame.add_data(
            {"animal_after_filtration": stats.get_animals_no(filtered_animals)}
        )
        self.stats_frame.add_data({"time_period": stats.get_period(filtered_animals)})

        self.stats_frame.add_data(
            {
                "min_label_no": [
                    [f"user_id : {index}", f"no: {value}"]
                    for index, value in stats.get_min_labels_no_after_filtration(
                        filtered_animals
                    ).items()
                ]
            }
        )

        self.stats_frame.add_data(
            {
                "min_records": [
                    [f"user_id : {index}", f"no: {value}"]
                    for index, value in stats.get_min_records_no_before_filtration(
                        self.data
                    ).items()
                ]
            }
        )

        self.stats_frame.add_data(
            {"avg_duration": stats.get_mean_periods(filtered_animals)}
        )
        self.stats_frame.add_data(
            {"min_duration": stats.get_min_periods(filtered_animals)}
        )
        self.stats_frame.add_data(
            {"max_duration": stats.get_max_periods(filtered_animals)}
        )
        self.stats_frame.add_data(
            {"overall_set_area": stats.get_overall_area(filtered_animals)}
        )
        self.stats_frame.add_data(
            {"average_set_area": stats.get_mean_area(filtered_animals)}
        )
        self.stats_frame.add_data(
            {"min_area": stats.get_min_area(filtered_animals)}
        )
        self.stats_frame.add_data(
            {"max_area": stats.get_max_area(filtered_animals)}
        )

        self.stats_frame.add_data(
            {
                "mean_label_no": stats.get_mean_labels_no_after_filtration(
                    filtered_animals
                )
            }
        )
        self.stats_frame.add_data(
            {"std_label_no": stats.get_std_labels_no_after_filtration(
                filtered_animals)
            }
        )
        self.stats_frame.add_data(
            {"mean_records": stats.get_mean_records_no_before_filtration(
                self.data)
            }
        )
        self.stats_frame.add_data(
            {"std_records": stats.get_std_records_no_before_filtration(
                self.data)
            }
        )
        self.stats_frame.add_data(
            {"std_duration": stats.get_std_periods(filtered_animals)}
        )
        self.stats_frame.add_data(
            {"std_area": stats.get_std_area(filtered_animals)}
        )

        return filtered_animals


    def _advenced_preprocessing(self):
        """
        Performs advanced preprocessing by filtering, filling missing data,
        and calculating distinct locations over time.

        Returns:
            tuple: Processed data, number of rows, data index, and unique
                locations count.
        """

        preproc = Prepocessing()
        filtrated_data = preproc.filter_by_quartiles(self.data)
        data = (
            filtrated_data.reset_index()
            .drop(columns=["Unnamed: 0", "geometry"])
            .drop_duplicates()
            .set_index("user_id")
        )
        filled_data = preproc.filing_data(data)
        nrows = int(filled_data.groupby(level=0).apply(lambda x: len(x)).min())
        n_data = np.arange(1, nrows + 1)
        S_data = [
            filled_data.groupby(level=0)["new_sum"].nth(x).mean() for x in range(nrows)
        ]

        return filled_data, nrows, n_data, S_data


    def process_file(self) -> None:
        """
        Main processing function that executes preprocessing,
        statistical analysis, and generates a PDF report with various
        scaling laws.
        """

        self.stats_frame.add_data({"animal": self.animal_name})

        filtered_animals = self._preprocess_data()
        filtered_animals.to_csv(
            os.path.join(self.output_dir, f"compressed_{self.animal_name}.csv")
        )
        filled_data, nrows, n_data, S_data = self._advenced_preprocessing()

        min_label_no = [
            [value]
            for index, value in Stats()
            .get_min_labels_no_after_filtration(filtered_animals)
            .items()
        ][0][0]

        min_records = [
            [value]
            for index, value in Stats()
            .get_min_records_no_before_filtration(self.data)
            .items()
        ][0][0]

        laws = Laws(
            pdf_object=self.pdf,
            stats_frame=self.stats_frame,
            output_path=self.output_dir_animal,
        )
        laws.visitation_frequency(filtered_animals, min_label_no)
        laws.distinct_locations_over_time(nrows, n_data, S_data)
        laws.msd_curve_split(filled_data)
        laws.rog_over_time(filtered_animals, min_records)
        self.pdf.add_page()
        laws.waiting_times(filtered_animals)
        laws.jump_lengths_distribution(filtered_animals)
        # laws.travel_times(filtered_animals)
        # laws.rog(filtered_animals)
        laws.msd_distribution(filtered_animals)
        # laws.return_time_distribution(filtered_animals)
        # laws.exploration_time(filtered_animals)
        laws.estimate_pnew(filled_data)

        pdf_path = os.path.join(
            self.output_dir_animal, f"{self.animal_name}.pdf"
        )
        self.pdf.output(pdf_path)

        self.stats_frame.add_record()
