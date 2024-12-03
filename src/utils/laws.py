from datetime import timedelta
from numpy import ndarray, size
import pandas as pd
import os
import logging
from humobi.structures.trajectory import TrajectoriesFrame
import scipy
import scipy.stats
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
import scipy.stats as scp_stats

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
        Computes a linear transformation: y = a * x * b.

        Parameters:
        x (array-like): Input values.
        a (float): Slope coefficient.
        b (float): Scaling coefficient.

        Returns:
        array-like: Transformed values.
        """
        x = x.astype(float)
        return a * x * b

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
    def cubic(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
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
        Computes a sigmoid curve: y = 1 / (1 + exp(a*x + b)).

        Parameters:
        x (array-like): Input values.
        a (float): Scaling factor.
        b (float): Offset factor.

        Returns:
        array-like: Transformed values.
        """
        x = x.astype(float)
        return 1 / (1 + np.exp(a * x + b))

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


class DistributionFitingTools:
    """
    A class containing tools for fitting distributions and models to data.
    """

    def __init__(self) -> None:
        self.curves = Curves()

    def _fit_distribution(self, data, distribution) -> float:
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

    def _calculate_akaike_weights(self, aic_values) -> np.ndarray:
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

    def _multiple_distributions(self, data) -> tuple:
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

    def model_choose(self, vals) -> tuple:
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

                y_pred = c(vals.index, *params)
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

        global_params = pd.DataFrame(columns=["curve", "weight", "param1", "param2"])
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
    def get_min_labels_no_after_filtration(data: TrajectoriesFrame) -> pd.Series:
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
        return unique_label_counts[unique_label_counts == min_unique_label_count]

    @staticmethod
    def get_min_records_no_before_filtration(data: TrajectoriesFrame) -> pd.Series:
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
    def get_mean_periods(data: TrajectoriesFrame) -> pd.Timedelta:
        """
        Get the mean period between start and end times.

        Args:
            data (TrajectoriesFrame): Input data with 'start'
                and 'end' columns.

        Returns:
            float: The mean period in days.
        """
        return (data["end"] - data["start"]).mean()  # type: ignore

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
        return (data["end"] - data["start"]).min()  # type: ignore

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
        return (data["end"] - data["start"]).max()  # type: ignore

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


class DataSetStats:

    def __init__(self, output_dir) -> None:
        self.output_dir = output_dir
        self.record = {}
        self.stats_frame = pd.DataFrame(
            columns=[
                "animal",
                "animal_no",
                "animal_after_filtration",
                "time_period",
                "min_label_no",
                "min_records",
                "avg_duration",
                "min_duration",
                "max_duration",
                "overall_set_area",
                "average_set_area",
                "min_area",
                "max_area",
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

    def add_data(self, data: dict) -> None:
        self.record.update(data)

    def add_record(self) -> None:
        self.stats_frame._append(self.record, ignore_index=True)  # type: ignore
        self.record = {}


class Prepocessing:

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_mean_points(data: TrajectoriesFrame) -> TrajectoriesFrame:
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

        return TrajectoriesFrame(geometry_df.sort_values("datetime").drop_duplicates())

    @staticmethod
    def set_start_stop_time(data: TrajectoriesFrame) -> TrajectoriesFrame:
        compressed = pd.DataFrame(
            start_end(data).reset_index()[
                ["user_id", "datetime", "labels", "lat", "lon", "date", "start", "end"]
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

        data.set_crs(base_csr)
        data.to_crs(target_crs)

        return data

    @staticmethod
    def filter_by_min_number(
        data: TrajectoriesFrame, min_labels_no: int = const.MIN_LABEL_NO
    ) -> TrajectoriesFrame:

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


class Laws:

    def __init__(self, pdf_object: FPDF, stats_dict: dict, output_path: str) -> None:
        self.pdf_object = pdf_object
        self.output_path = output_path


class ScalingLawsCalc:

    def __init__(
        self,
        data: TrajectoriesFrame,
        data_name: str,
        output_dir: str,
        stats_frame: DataSetStats,
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

        print("RAW ANIMAL NO:", stats.get_animals_no(self.data))
        print("FILTRED ANIMAL NO:", stats.get_animals_no(filtered_animals))

        print("RAW ANIMAL PERIOD:", stats.get_period(filtered_animals))
        print("FILTRED ANIMAL PERIOD:", stats.get_period(filtered_animals))

        print(
            "MIN RECORDS NO BEF FILTRATION :",
            stats.get_min_records_no_before_filtration(self.data),
        )
        print(
            "MIN LABELS NO AFTER FILTRATION :",
            stats.get_min_labels_no_after_filtration(filtered_animals),
        )

        # FIXME: choose data for compressed csv and next step of calculations
        return compressed_points, filtered_animals

    def process_file(self) -> None:

        compressed_points, filtered_animals = self._preprocess_data()
