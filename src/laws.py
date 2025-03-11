
from typing_extensions import Buffer
from numpy import ndarray
import pandas as pd
import os
import logging
from humobi.structures.trajectory import TrajectoriesFrame
import scipy
import scipy.stats
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
from humobi.measures.individual import visitation_frequency, jump_lengths, distinct_locations_over_time, radius_of_gyration, mean_square_displacement, num_of_distinct_locations
from humobi.tools.processing import rowwise_average, convert_to_distribution, start_end
from constans import const
import scipy.stats as scp_stats
from distfit import distfit
from tqdm import tqdm
from io import BytesIO
from math import log
from scipy.stats import wasserstein_distance
import ruptures as rpt


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
        return A * n ** B


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

        return (data.groupby("user_id")["end"].max() - data.groupby("user_id")["start"].min()).mean()  # type: ignore

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
        return (data.groupby("user_id")["end"].max() - data.groupby("user_id")["start"].min()).min()  # type: ignore

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
        return (data.groupby("user_id")["end"].max() - data.groupby("user_id")["start"].min()).max()  # type: ignore

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
        self.stats_set = pd.DataFrame(
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
        self.stats_set = pd.concat([self.stats_set, pd.DataFrame([self.record])], ignore_index=True)  # type: ignore
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
        data_frame = data.copy().set_crs(base_csr)  # type: ignore

        return data_frame.to_crs(target_crs)

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

    @staticmethod
    def filing_data(data:pd.DataFrame) -> pd.DataFrame:

        def longest_visited_row(groupa):
            """
            Returns the row where the individual spent the longest time in an interval.
            """
            if groupa.empty:
                return pd.Series(dtype=object)
            max_label = groupa.groupby('labels')['duration'].sum().idxmax()
            row = groupa[groupa['labels'] == max_label].iloc[0]

            return row.T

        to_conca = {}
        for uid, group in data.groupby(level=0):
            group = group[~group['datetime'].duplicated()]
            if len(group.labels.unique()) < 2:
                continue
            group.set_index('datetime', inplace=True)
            group['duration'] = (group.index.to_series().shift(-1) - group.index).dt.total_seconds()
            group['duration'].fillna(3600, inplace=True)

            group_resampled = group.resample('1H').apply(longest_visited_row).unstack()
            if group_resampled.index.nlevels > 1:
                group_resampled = group.resample('1H').apply(longest_visited_row)
            group_resampled = group_resampled.resample('1H').first()
            group_resampled = group_resampled.ffill().bfill()
            to_conca[uid] = group_resampled

        return pd.DataFrame(pd.concat(to_conca))

class Flexation:

    def __init__(self) -> None:
        pass

    def _calculate_penalty(self, data):
        if const.FLEXATION_POINTS_SENSITIVITY == "Low":
            return 6 * log(len(data))
        elif const.FLEXATION_POINTS_SENSITIVITY == "Medium":
            return 3 * log(len(data))
        elif const.FLEXATION_POINTS_SENSITIVITY == "High":
            return 1.5 * log(len(data))

    def _calc_main_model_wasser(
        self, model_obj: distfit, data: ndarray, flexation_point: int
    ) -> float:

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

    def _fit_mixed_models(self, data: ndarray, flexation_points: list) -> pd.DataFrame:
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
                left_model =distfit(distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform', 'loggamma','truncexpon','truncnorm','truncpareto','powerlaw'],stats="wasserstein")
                right_model =distfit(distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform', 'loggamma','truncexpon','truncnorm','truncpareto','powerlaw'],stats="wasserstein")

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
        penalty = self._calculate_penalty(data)
        model = rpt.Pelt(model="rbf").fit(data)
        break_points_indx = model.predict(pen=penalty)
        return [data[i - 1] for i in break_points_indx]

    def find_distributions(self, model: distfit, data: ndarray):
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

                left_model = distfit(distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform', 'loggamma','truncexpon','truncnorm','truncpareto','powerlaw'],stats="wasserstein")
                right_model = distfit(distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform', 'loggamma','truncexpon','truncnorm','truncpareto','powerlaw'],stats="wasserstein")

                left_model.fit_transform(left_set)
                right_model.fit_transform(right_set)
                return left_model, right_model, left_set, right_set, best_point


class Laws:

    def __init__(self, pdf_object: FPDF, stats_frame: DataSetStats, output_path: str):
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
        y_position:float = None
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
        if y_position== None:
            y_position = self.pdf_object.get_y()
        else:
            y_position=y_position

        try:
            self.pdf_object.image(
                plot_obj, x=x_position, y=y_position, w=image_width, h=image_height
            )
            self.pdf_object.set_y(y_position + image_height + 10)
            plot_obj.close()
        except Exception as e:
            raise RuntimeError(f"Failed to add plot to PDF: {e}")

    def _add_pdf_curves_table(self, data,x_offset=10,y_offset=None):
        if y_offset== None:
            y_offset = self.pdf_object.get_y()
        else:
            y_offset=y_offset
        self.pdf_object.set_xy(x_offset, y_offset)

        self.pdf_object.set_font("Arial", size=7)
        col_width = 25
        self.pdf_object.set_font("Arial", style="B", size=6)
        self.pdf_object.cell(col_width, 3, "Curve", border='TB', align="C")
        self.pdf_object.cell(col_width, 3, "Weight", border='TB', align="C")
        self.pdf_object.cell(col_width, 3, "Param 1", border='TB', align="C")
        self.pdf_object.cell(col_width, 3, "Param 2", border='TB', align="C")
        self.pdf_object.ln()
        self.pdf_object.set_font("Arial", size=6)

        for index, row in data.iterrows():
            self.pdf_object.cell(col_width, 3, row["curve"], border=0, align="C")
            self.pdf_object.cell(col_width, 3, str(round(row["weight"],10)), border=0, align="C")
            self.pdf_object.cell(col_width, 3, str(round(row["param1"],10)), border=0, align="C")
            self.pdf_object.cell(col_width, 3, str(round(row["param2"],10)), border=0, align="C")
            self.pdf_object.ln()


        self.pdf_object.cell(col_width*4, 0, "", border="T")
        self.pdf_object.ln(1)

    def _add_pdf_distribution_table(self, data):
        self.pdf_object.set_font("Arial", style="B", size=6)
        self.pdf_object.cell(35, 3, "Distribution", border='TB', align="C")
        self.pdf_object.cell(50, 3, "Score", border='TB', align="C")
        self.pdf_object.cell(100, 3, "Params", border='TB', align="C")
        self.pdf_object.set_font("Arial", size=6)
        self.pdf_object.ln()

        for index, row in data.iterrows():
            try:
                self.pdf_object.cell(35, 3, row["name"], border=0, align="C")
                self.pdf_object.cell(50, 3, str(round(row["score"],15)), border=0, align="C")
                self.pdf_object.cell(100, 3, str(tuple(round(x, 5) for x in row["params"])).replace("(","").replace(")",""), border=0, align="C")
                # self.pdf_object.cell(40, 5, str(row["params"]), border=1, align="C")
                self.pdf_object.ln()
            except:
                pass
        self.pdf_object.cell(185, 0, "", border="T")
        self.pdf_object.ln(1)

    def _plot_curve(self, func_name, plot_data, y_pred, labels, exp_y_pred=None):
        buffer = BytesIO()

        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 4.5))

        if exp_y_pred is not None and exp_y_pred.size > 0:
            plt.plot(plot_data.index, y_pred, c="k", linestyle="--", label="Sigmoid")
            plt.plot(plot_data.index, exp_y_pred, c="r", linestyle="-.", label="Expon neg")

        else:
            plt.plot(plot_data.index, y_pred, c="k", linestyle="--", label="Fitted curve")

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
        buffer_plot_distribution = BytesIO()
        buffer_plot_model = BytesIO()

        measure_type = (
            measure_type.replace("_", " ").replace("distribution", "").capitalize()
        )

        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 6))
        plt.hist(values, color='darkturquoise',bins=100, density=True)
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
            emp_properties={"color": "darkturquoise","linewidth": 0, "marker": "o"},
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

    def _plot_P_new(self, rho_hat,gamma_hat,nrows):
        buffer = BytesIO()
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 4.5))
        plt.plot(np.arange(1, nrows), [rho_hat * x ** (-gamma_hat) for x in range(1, nrows)], label="Estimated", color="darkturquoise")
        plt.plot(np.arange(1, nrows), [0.6 * x ** (-0.21) for x in range(1, nrows)], label="Paper", color="black")
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
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        return buffer

    def _plot_MSD_split(self):
        pass

    def _plot_DLOT_split(self):
        pass

    def _plot_Ploglog(self):
        pass

    def log_curve_fitting_resluts(func):
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

            func_name = func_name.replace('_',' ').split(' ')
            func_name[0] = func_name[0].capitalize()
            self.pdf_object.set_font("Arial", "B", size=8)
            self._add_pdf_cell(f"{' '.join(func_name)}")
            self.pdf_object.set_font("Arial", size=7)
            self._add_pdf_cell(
                f'Best fit: {best_fit} with Param 1: {filtered_df["param1"].values[0]}, Param 2: {filtered_df["param2"].values[0]}'
            )

            y_position_global = float(self.pdf_object.get_y())
            self._add_pdf_curves_table(param_frame,x_offset=10,y_offset=y_position_global+13)
            self._add_pdf_plot(plot_obj=plot_obj,image_width=80, image_height=45, x_position= 125, y_position = y_position_global)

        return wrapper

    def log_distribution_fitting_resluts(func):
        def wrapper(self, *args, **kwargs):
            results = func(self, *args, **kwargs)

            self.stats_frame.add_data({results[0]: results[1].model["name"]})
            self.stats_frame.add_data(
                {f"{results[0]}_params": results[1].model["params"]}
            )

            func_name = results[0].replace('_',' ').split(' ')
            func_name[0] = func_name[0].capitalize()
            self.pdf_object.set_font("Arial", "B", size=8)
            self._add_pdf_cell(f"{' '.join(func_name)}")
            self.pdf_object.set_font("Arial", size=7)
            self._add_pdf_cell(
                f'Best fit: {results[1].model["name"]} with params: {results[1].model["params"]}'
            )

            self._add_pdf_distribution_table(results[1].summary[["name", "score",'params']])
            y_position_global = float(self.pdf_object.get_y())
            self._add_pdf_plot(results[2], 80, 60,x_position= 10, y_position = y_position_global)
            self._add_pdf_plot(results[3], 80, 60,x_position= 110, y_position = y_position_global)

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
                self._add_pdf_distribution_table(results[4][1].summary[["name", "score",'params']])
                plot_distribution, plot_models = self._plot_double_distribution(
                    results[4][0],
                    results[4][1],
                    results[4][2],
                    results[4][3],
                    results[4][4],
                    results[0],
                )
                y_position_global = float(self.pdf_object.get_y())
                self._add_pdf_plot(plot_distribution, 80, 60,x_position= 10, y_position = y_position_global)
                self._add_pdf_plot(plot_models, 80, 60,x_position= 110, y_position = y_position_global)
            self.pdf_object.add_page()

        return wrapper

    def log_pnew_estimation(func):
        def wrapper(self, *args, **kwargs):
            rho_hat, gamma_hat, A_fit, B_fit, nrows = func(
                self, *args, **kwargs
            )  # type: ignore
            self.pdf_object.set_font("Arial", "B", size=8)
            self._add_pdf_cell("Pnew estimation")
            y_position_global = float(self.pdf_object.get_y())
            self.pdf_object.ln()
            self.pdf_object.ln()
            self.pdf_object.set_font("Arial", size=7)
            self._add_pdf_cell(
                f"A_fit  = {A_fit:.4f} (paper: {const.A_FIT})")
            self._add_pdf_cell(
                f"B_fit  = {B_fit:.4f} (paper: {const.B_FIT})")
            self.pdf_object.set_font("Arial", "B",size=7)
            self._add_pdf_cell(
                f"gamma  = {gamma_hat:.4f} (paper: {const.GAMMA})")
            self._add_pdf_cell(
                f"rho  = {rho_hat:.4f} (paper: {const.RHO})")
            plot_obj = self._plot_P_new(rho_hat,gamma_hat,nrows)
            self._add_pdf_plot(plot_obj=plot_obj,image_width=80, image_height=45, x_position= 110, y_position = y_position_global)

        return wrapper

    def check_curve_fit(func):
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
            elif(
                model.model["name"] == 'pareto' and model.model['params'][0] > 2
            ):
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
            elif(
                model.model["name"] == 'lognorm' and model.model['params'][0] > 2
            ):
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
        vf = visitation_frequency(data)
        avg_vf = rowwise_average(vf, row_count=min_labels_no)
        avg_vf.index += 1
        vf.groupby(level=0).size().median()
        avg_vf = avg_vf[~avg_vf.isna()]

        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            self.curve_fitting.model_choose(avg_vf)
        )

        return best_fit, global_params, y_pred, expon_y_pred, avg_vf, ["Rank","f"]

    @log_curve_fitting_resluts
    @check_curve_fit
    def distinct_locations_over_time(
        self, data: TrajectoriesFrame, min_labels_no: int
    ) -> tuple:

        dlot = distinct_locations_over_time(data,reaggregate = True, resolution = '1H')
        avg_dlot = rowwise_average(dlot, row_count=min_labels_no)
        avg_dlot.index += 1
        dlot.groupby(level=0).size().median()
        avg_dlot = avg_dlot[~avg_dlot.isna()]

        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(avg_dlot)
        )

        return best_fit, global_params, y_pred, expon_y_pred, avg_dlot, ["t","S(t)"]

    @log_distribution_fitting_resluts
    @check_distribution_fit
    def jump_lengths_distribution(self, data: TrajectoriesFrame) -> tuple:
        jl = jump_lengths(data)
        jl = jl[jl != 0]
        jl_dist = convert_to_distribution(jl, num_of_classes=20)

        # Fit to find the best theoretical distribution
        jl = jl[~jl.isna()]
        model = distfit(distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform', 'loggamma','truncexpon','truncnorm','truncpareto','powerlaw'],stats="wasserstein")
        model.fit_transform(jl.values)

        return model, jl

    @log_distribution_fitting_resluts
    @check_distribution_fit
    def waiting_times(self, data: TrajectoriesFrame) -> tuple:
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
        model = distfit(distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform', 'loggamma','truncexpon','truncnorm','truncpareto','powerlaw'],stats="wasserstein")
        model.fit_transform(wt.values)

        return model, wt

    @log_distribution_fitting_resluts
    @check_distribution_fit
    def travel_times(self, data: TrajectoriesFrame) -> tuple:
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
        model = distfit(distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform', 'loggamma','truncexpon','truncnorm','truncpareto','powerlaw'],stats="wasserstein")
        model.fit_transform(tt.values)

        return model, tt

    @log_distribution_fitting_resluts
    @check_distribution_fit
    def rog(self, data: TrajectoriesFrame) -> tuple:
        rog = radius_of_gyration(data, time_evolution=False)

        # Fit to find the best theoretical distribution
        model = distfit(distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform', 'loggamma','truncexpon','truncnorm','truncpareto','powerlaw'],stats="wasserstein")
        model.fit_transform(rog.values)

        return model, rog

    @log_curve_fitting_resluts
    @check_curve_fit
    def rog_over_time(self, data: TrajectoriesFrame, min_records_no: int) -> tuple:
        rog = radius_of_gyration(data, time_evolution=True)
        avg_rog = rowwise_average(rog,row_count=min_records_no)
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
        msd = mean_square_displacement(data, time_evolution=False, from_center=True)
        # Fit to find the best theoretical distribution
        model = distfit(distr=['norm', 'expon', 'pareto', 'dweibull', 't', 'genextreme', 'gamma', 'lognorm', 'beta', 'uniform', 'loggamma','truncexpon','truncnorm','truncpareto','powerlaw'],stats="wasserstein")
        model.fit_transform(msd.values)

        return model, msd

    @log_curve_fitting_resluts
    @check_curve_fit
    def msd_curve(self, data: TrajectoriesFrame, min_records_no: int) -> tuple:
        msd = mean_square_displacement(data, time_evolution=True, from_center=False)
        avg_msd = rowwise_average(msd, row_count=min_records_no)
        avg_msd = avg_msd[~avg_msd.isna()]
        # model selection
        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(avg_msd)
        )

        return best_fit, global_params, y_pred, expon_y_pred, avg_msd, ["t","MSD"]

    @log_distribution_fitting_resluts
    @check_distribution_fit
    def return_time_distribution(self, data: TrajectoriesFrame) -> tuple:
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
            )  # type: ignore
            result = result.dt.total_seconds()
            to_concat[uid] = result

        rt = pd.concat(to_concat)
        rt = rt.reset_index(level=1, drop=True)
        rt = rt[rt != 0]
        rt = pd.concat(to_concat)
        rt = rt[rt != 0]

        # Fit to find the best theoretical distribution
        model = distfit(stats="wasserstein")
        model.fit_transform(rt.values)

        return model, rt

    @log_distribution_fitting_resluts
    @check_distribution_fit
    def exploration_time(self, data: TrajectoriesFrame) -> tuple:
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
            )  # type: ignore
            if result.size == 0:
                continue
            result = result.dt.total_seconds()
            to_concat[uid] = result

        et = pd.concat(to_concat)
        et = et.reset_index(level=1, drop=True)
        et = et[et != 0]

        # Fit to find the best theoretical distribution
        model = distfit(stats="wasserstein")
        model.fit_transform(et.values)

        return model, et

    def msd_curve_split(self, data):

        pass

    @log_curve_fitting_resluts
    @check_curve_fit
    def distinct_locations_over_time_split(self, data):
        nrows = int(data.groupby(level=0).apply(lambda x: len(x)).min())
        n_data = np.arange(1, nrows + 1)
        S_data = [data.groupby(level=0)['new_sum'].nth(x).mean() for x in range(nrows)]
        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(pd.Series(S_data))
        )
        return best_fit, global_params, y_pred, expon_y_pred, pd.Series(S_data), ["t","S(t)"]

    @log_pnew_estimation
    def estimate_pnew(self, data:pd.DataFrame) -> tuple:
        """
        Estimate parameters (rho, gamma) for the new place probability
        function P_new(S) = rho * S^(-gamma).

        Parameters:


        Returns:
            rho_hat (float): Estimated rho parameter.
            gamma_hat (float): Estimated gamma parameter.
        """
        nrows = int(data.groupby(level=0).apply(lambda x: len(x)).min())
        n_data = np.arange(1, nrows + 1)
        S_data = [data.groupby(level=0)['new_sum'].nth(x).mean() for x in range(nrows)]
        # Fit power-law model to S(n) = A * n^B
        popt, pcov = curve_fit(Curves.power_law, n_data, S_data, p0=const.PNEW_P0)
        A_fit, B_fit = popt

        # Compute gamma and rho based on theoretical model
        gamma_hat = 1.0 / B_fit - 1.0
        rho_hat = (A_fit ** (gamma_hat + 1.0)) / (gamma_hat + 1.0)
        return rho_hat, gamma_hat, A_fit, B_fit, nrows

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
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.set_font("Arial", "B", size=12)
        self.pdf.cell(200, 10, text=f"{self.animal_name.replace('_',' ')}", ln=True, align="C")
        self.pdf.set_font("Arial", size=9)
        self.pdf.ln(5)

    def _preprocess_data(self) -> TrajectoriesFrame:
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
        self.stats_frame.add_data({"min_area": stats.get_min_area(filtered_animals)})
        self.stats_frame.add_data({"max_area": stats.get_max_area(filtered_animals)})

        # FIXME: choose data for compressed csv and next step of calculations
        return filtered_animals

    def _advenced_preprocessing(self):
        preproc = Prepocessing()
        data = self.data.reset_index().drop(columns=['Unnamed: 0']).drop_duplicates().set_index('user_id')
        filled_data = preproc.filing_data(data)

        filled_data['is_new'] = filled_data.groupby(level=0, group_keys=False).apply(lambda x: ~x.duplicated(keep='first'))
        filled_data['new_sum'] = filled_data.groupby(level=0).apply(lambda x: x.is_new.cumsum()).droplevel(1)

        return filled_data

    def process_file(self) -> None:

        self.stats_frame.add_data({"animal": self.animal_name})

        filtered_animals = self._preprocess_data()
        filtered_animals.to_csv(os.path.join(self.output_dir,f'compressed_{self.animal_name}.csv'))

        filled_animals = self._advenced_preprocessing()

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
        laws.distinct_locations_over_time_split(filled_animals)
        # laws.distinct_locations_over_time(filtered_animals, min_label_no)
        laws.msd_curve(filtered_animals, min_records)
        laws.rog_over_time(filtered_animals, min_records)
        self.pdf.add_page()
        laws.waiting_times(filtered_animals)
        laws.jump_lengths_distribution(filtered_animals)
        # laws.travel_times(filtered_animals)
        # laws.rog(filtered_animals)
        laws.msd_distribution(filtered_animals)
        # laws.return_time_distribution(filtered_animals)
        # laws.exploration_time(filtered_animals)
        laws.estimate_pnew(filled_animals)

        pdf_path = os.path.join(self.output_dir_animal, f"{self.animal_name}.pdf")
        self.pdf.output(pdf_path)

        self.stats_frame.add_record()

