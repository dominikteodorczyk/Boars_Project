import inspect
import traceback
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from distfit import distfit
from humobi.structures.trajectory import TrajectoriesFrame
from humobi.measures.individual import *
from humobi.tools.processing import *
from humobi.tools.user_statistics import *
from src.measures.distribution import *
from src.measures.stats import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )


class Measures:
    """
    A class to calculate various statistical measures related
    to animal trajectories.

    Attributes:
    ----------
    clear_data : TrajectoriesFrame
        The clear data used for calculations.
    visitation_frequency_chart : any type or None
        Chart data for visitation frequency.
    calc_type : str
        The type of calculation to be performed ('jupiter' or other).
    stats : MeasurmentsStatistics
        An object to store the statistical results.
    """
    def __init__(
            self,
            clear_data: TrajectoriesFrame,
            calc_type: str,
            min_label_no:int,
            min_records_no:int
            ) -> None:

        self.clear_data = clear_data
        self.visitatio_frequency_chart = None
        self.calc_type = calc_type
        self.min_label_no = min_label_no
        self.min_records_no = min_records_no
        self.stats = MeasurmentsStatistics()

    def curve_plt(self, rowwise_avg, y_pred, y_label, x_label):

        # Display of data fitted curve
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(rowwise_avg.index, y_pred, c="k")
        plt.scatter(rowwise_avg.index, rowwise_avg)
        plt.loglog()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()


    def distribution_plt(
            self,
            model,
            values,
            measure_type:str = 'Values'):

        measure_type = measure_type.replace(
            '_',' '
            ).replace(
                'distribution',''
                ).capitalize()

        # Display of data distribution
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 5))
        plt.hist(values, bins=100, density=True)
        plt.loglog()

        # Display of matched distribution model
        plt.figure(figsize=(12, 5))
        model.plot(
            pdf_properties={
                "color": "#472D30",
                "linewidth": 4,
                "linestyle": "--"
                },
            bar_properties=None,
            cii_properties=None,
            emp_properties={
                "color": "#E26D5C",
                "linewidth": 0,
                "marker": "o"
                },
            figsize=(8, 5),
        )
        plt.xlabel(measure_type)
        plt.loglog()


    def visitation_frequency(self):
        """
        Calculates and stores the visitation frequency statistics.
        Plots the visitation frequency if the calc_type is 'jupiter'.
        """
        vf = visitation_frequency(self.clear_data)
        avg_vf = rowwise_average(vf, row_count=self.min_label_no)
        avg_vf.index = avg_vf.index + 1
        vf.groupby(level=0).size().median()
        avg_vf = avg_vf[~avg_vf.isna()]

        # model selection
        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(avg_vf)
        )

        # writing statistics
        self.stats.visitation_frequency = [
            global_params,
            best_fit,
            best_fit_params
            ]

        # ploting
        if self.calc_type == "jupiter":
            self.curve_plt(
                rowwise_avg = avg_vf,
                y_pred = y_pred,
                y_label = 'f',
                x_label = 'Rank'
                )
            if best_fit == 'sigmoid':
                self.curve_plt(
                    rowwise_avg=avg_vf,
                    y_pred= expon_y_pred,
                    y_label= 'f',
                    x_label= 'Rank'
                )



    def distinct_locations_over_time(self):
        """
        Calculates and stores the distinct locations over time statistics.
        Plots the distinct locations over time if the calc_type is 'jupiter'.
        """

        dlot = distinct_locations_over_time(self.clear_data)
        avg_dlot = rowwise_average(dlot, row_count=self.min_label_no)
        avg_dlot.index += 1
        dlot.groupby(level=0).size().median()
        avg_dlot = avg_dlot[~avg_dlot.isna()]

        # model selection
        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(avg_dlot)
        )

        # writing statistics
        self.stats.distinct_locations_over_time = [
            global_params,
            best_fit,
            best_fit_params,
        ]

        # ploting
        if self.calc_type == "jupiter":
            self.curve_plt(
                rowwise_avg = avg_dlot,
                y_pred = y_pred,
                y_label = 'S(t)',
                x_label = 't'
                )
            if best_fit == 'sigmoid':
                self.curve_plt(
                    rowwise_avg=avg_dlot,
                    y_pred= expon_y_pred,
                    y_label= 'S(t)',
                    x_label= 't'
                )


    def jump_lengths_distribution(self):
        """
        Calculates and stores the jump lengths distribution statistics.
        Plots the jump lengths distribution if the calc_type is 'jupiter'.
        """

        jl = jump_lengths(self.clear_data)
        jl = jl[jl != 0]
        jl_dist = convert_to_distribution(jl, num_of_classes=20)

        # Fit to find the best theoretical distribution
        jl = jl[~jl.isna()]
        model = distfit(stats="wasserstein")
        model.fit_transform(jl.values)
        logging.info(f'Best fit: {model.model["name"]},{ model.model["params"]}')

        # writing statistics
        self.stats.jump_lengths_distribution = [
            model.summary[["name", "score", "loc", "scale", "arg"]],
            model.model["name"],
            model.model["params"],
        ]

        # ploting
        if self.calc_type == "jupiter":

            self.distribution_plt(
                model = model,
                values = jl,
                measure_type = inspect.currentframe().f_code.co_name
                )


    def waiting_times(self):
        """
        Calculates and stores the waiting times statistics.
        Plots the waiting times distribution if the calc_type is 'jupiter'.
        """
        try:
            wt = self.clear_data.groupby(level=0).apply(
                lambda x: (x.end - x.start).dt.total_seconds()
            )
        except:
            wt = (self.clear_data['end'] - self.clear_data['start']).dt.total_seconds()

        wt = wt[~wt.isna()]
        wt = wt[wt != 0]

        # Fit to find the best theoretical distribution
        model = distfit(stats="wasserstein")
        model.fit_transform(wt.values)
        logging.info(f'Best fit: {model.model["name"]},{ model.model["params"]}')

        # writing statistics
        self.stats.waiting_times = [
            model.summary[["name", "score", "loc", "scale", "arg"]],
            model.model["name"],
            model.model["params"],
        ]

        # ploting
        if self.calc_type == "jupiter":
            self.distribution_plt(
                model=model,
                values=wt,
                measure_type= 'Waiting Time (seconds)'
            )


    def travel_times(self):
        """
        Calculates and stores the travel times statistics.
        Plots the travel times distribution if the calc_type is 'jupiter'.
        """
        try:
            tt = (
                self.clear_data.groupby(level=0)
                .progress_apply(lambda x: x.shift(-1).start - x.end)
                .reset_index(level=[1, 2], drop=True)
            )
        except:
            shifted_start = self.clear_data['start'].shift(-1)
            tt = shifted_start - self.clear_data['end']
            tt = tt.reset_index(drop=True)

        tt = tt.dt.total_seconds()
        tt = tt[~tt.isna()]

        # Fit to find the best theoretical distribution
        model = distfit(stats="wasserstein")
        model.fit_transform(tt.values)
        logging.info(f'Best fit: {model.model["name"]},{ model.model["params"]}')

        # writing statistics
        self.stats.travel_times = [
            model.summary[["name", "score", "loc", "scale", "arg"]],
            model.model["name"],
            model.model["params"],
        ]

        # ploting
        if self.calc_type == "jupiter":
            self.distribution_plt(
                model=model,
                values=tt
            )


    def rog(self):
        """
        Calculates and stores the radius of gyration statistics.
        Plots the radius of gyration distribution if the
        calc_type is 'jupiter'.
        """
        rog = radius_of_gyration(self.clear_data, time_evolution=False)

        # Fit to find the best theoretical distribution
        model = distfit(stats="RSS")
        model.fit_transform(rog.values)
        logging.info(f'Best fit: {model.model["name"]},{ model.model["params"]}')

        # writing statistics
        self.stats.rog = [
            model.summary[["name", "score", "loc", "scale", "arg"]],
            model.model["name"],
            model.model["params"],
        ]

        # ploting
        if self.calc_type == "jupiter":
            self.distribution_plt(
                model=model,
                values=rog,
            )


    def rog_over_time(self):
        """
        Calculates and stores the radius of gyration over time statistics.
        Plots the radius of gyration over time if the calc_type is 'jupiter'.
        """

        rog = radius_of_gyration(self.clear_data, time_evolution=True)
        avg_rog = rowwise_average(rog)


        # model selection
        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(avg_rog)
        )

        # writing statistics
        self.stats.rog_over_time = [global_params, best_fit, best_fit_params]

        # ploting
        if self.calc_type == "jupiter":
            plt.scatter(np.arange(avg_rog.size), avg_rog.values)

            self.curve_plt(
                rowwise_avg=avg_rog,
                y_pred= y_pred,
                y_label='',
                x_label= 'Values'
                )
            if best_fit == 'sigmoid':
                self.curve_plt(
                    rowwise_avg=avg_rog,
                    y_pred= expon_y_pred,
                    y_label= '',
                    x_label= 'Values'
                )

    def msd_distribution(self):
        """
        Calculates and stores the mean square displacement (MSD)
        distribution statistics. Plots the MSD distribution if
        the calc_type is 'jupiter'.
        """
        msd = mean_square_displacement(
            self.clear_data, time_evolution=False, from_center=True
        )

        # Fit to find the best theoretical distribution
        model = distfit(stats="wasserstein")
        model.fit_transform(msd.values)
        logging.info(f'Best fit: {model.model["name"]},{ model.model["params"]}')

        # writing statistics
        self.stats.msd_distribution = [
            model.summary[["name", "score", "loc", "scale", "arg"]],
            model.model["name"],
            model.model["params"],
        ]

        # ploting
        if self.calc_type == "jupiter":
            self.distribution_plt(
                model=model,
                values=msd
            )

    def msd_curve(self):
        """
        Calculates and stores the mean square displacement (MSD)
        curve statistics. Plots the MSD curve if the
        calc_type is 'jupiter'.
        """
        msd = mean_square_displacement(
            self.clear_data,
            time_evolution=True,
            from_center=False
        )
        avg_msd = rowwise_average(msd)

        # model selection
        y_pred, best_fit, best_fit_params, global_params, expon_y_pred = (
            DistributionFitingTools().model_choose(avg_msd)
        )

        # writing statistics
        self.stats.msd_curve = [global_params, best_fit, best_fit_params]

        # ploting
        if self.calc_type == "jupiter":
            self.curve_plt(
                rowwise_avg=avg_msd,
                y_pred= y_pred,
                y_label= 'MSD',
                x_label= 't'
            )
            if best_fit == 'sigmoid':
                self.curve_plt(
                    rowwise_avg=avg_msd,
                    y_pred= expon_y_pred,
                    y_label= 'MSD',
                    x_label= 't'
                )


    def return_time_distribution(self):
        """
        Calculates and stores the return time distribution statistics.
        Plots the return time distribution if the calc_type is 'jupiter'.
        """

        to_concat = {}
        for uid, vals in tqdm(
            self.clear_data.groupby(level=0),
            total=len(pd.unique(self.clear_data.index.get_level_values(0))),
        ):
            vals = vals.sort_index()[
                [
                "labels",
                "start",
                "end"]
                ]

            vals["new_place"] = ~vals["labels"].duplicated(keep="first")
            vals["islands"] = vals["new_place"] * (
                (vals["new_place"] != vals["new_place"].shift(1)).cumsum()
            )
            vals["islands_reach"] = vals["islands"].shift()
            vals["islands"] = vals[["islands", "islands_reach"]].max(axis=1)

            vals = vals.drop("islands_reach", axis=1)
            vals = vals[vals.islands > 0]

            result = vals.groupby("islands").apply(
                lambda x: x.iloc[-1].start - x.iloc[0].start if len(x) > 0 else None
            )
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
        logging.info(f'Best fit: {model.model["name"]},{ model.model["params"]}')

        # writing statistics
        self.stats.return_time_distribution = [
            model.summary[["name", "score", "loc", "scale", "arg"]],
            model.model["name"],
            model.model["params"],
        ]

        # ploting
        if self.calc_type == "jupiter":
            self.distribution_plt(
                model = model,
                values=rt
            )


    def exploration_time(self):
        """
        Calculates and stores the exploration time statistics.
        Plots the exploration time distribution if the calc_type is 'jupiter'.
        """

        to_concat = {}
        for uid, vals in tqdm(
            self.clear_data.groupby(level=0),
            total=len(
                pd.unique(
                    self.clear_data.index.get_level_values(0)
                    )
                ),
        ):
            vals = vals.sort_index()[
                [
                    "labels",
                    "start",
                    "end"
                    ]
                ]

            vals["old_place"] = vals["labels"].duplicated(keep="first")
            vals["islands"] = vals["old_place"] * (
                (vals["old_place"] != vals["old_place"].shift(1)).cumsum()
            )
            vals["islands_reach"] = vals["islands"].shift()
            vals["islands"] = vals[["islands", "islands_reach"]].max(axis=1)

            vals = vals.drop("islands_reach", axis=1)
            vals = vals[vals.islands > 0]

            result = vals.groupby("islands").apply(
                lambda x: x.iloc[-1].start - x.iloc[0].start if len(x) > 0 else None
            )
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

        # writing statistics
        self.stats.exploration_time = [
            model.summary[["name", "score", "loc", "scale", "arg"]],
            model.model["name"],
            model.model["params"],
        ]

        #ploting
        if self.calc_type == "jupiter":
            self.distribution_plt(
                model=model,
                values=et
            )


    def calculate_all(self):
        """
        It performs incorrectly in the notebook files.
        Recommended for quick calculations in terms of statistics
        (separate extraction of statistics from the statistics variable)
        """
        if self.clear_data.shape[0] != 0:
            try:
                self.visitation_frequency()
                self.distinct_locations_over_time()
                self.jump_lengths_distribution()
                self.waiting_times()
                self.travel_times()
                self.rog()
                self.rog_over_time()
                self.msd_distribution()
                self.msd_curve()
                self.return_time_distribution()
                self.exploration_time()

            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)
                error_traceback = traceback.format_exc()
                logging.error(f"Error type: {error_type}")
                logging.error(f"Error message: {error_message}")
                logging.error(f"Error traceback:\n{error_traceback}")
                pass
        else:
            logging.info('No data after filtration')