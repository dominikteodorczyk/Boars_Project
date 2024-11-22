from typing import final
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
from io import BytesIO
sns.set_style("whitegrid")
from infostop import Infostop
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import gc
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

class DataAnalystInfostop:
    """
    A class responsible for generating data analysis reports in PDF format.

    Parameters
    ----------
    pdf_object : FPDF
        PDF instance to which the analysis results will be added.
    """
    def __init__(self, pdf_object, output_path) -> None:
        """
        Initializes the DataAnalystInfostop with a given PDF instance.

        Parameters
        ----------
        pdf_object : FPDF
            PDF instance where analysis results will be written.
        """
        self.pdf_object = pdf_object
        self.output_path = output_path


    def generate_raport(self, data, data_analyst_no: int):
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
        self._add_records_per_time_frame(data, data_analyst_no, '1D')
        self.pdf_object.add_page()

        self._add_records_per_time_frame(data, data_analyst_no, '1H')
        self._add_trajctories_duration(data, data_analyst_no, '1H')
        self._add_no_of_consecutive_records(data, data_analyst_no, '1H')
        self.pdf_object.add_page()

        self._add_no_of_consecutive_records(data, data_analyst_no, '1D')
        self._add_average_temporal_resolution(data, data_analyst_no)
        self.pdf_object.add_page()


    def _add_pdf_cell(self,txt_to_add:str):
        """
        Adds a single line of text to the PDF.

        Parameters
        ----------
        txt_to_add : str
            Text to be added to the PDF document.
        """
        self.pdf_object.cell(200, 5, txt=txt_to_add, ln=True, align='L')

    def _add_pdf_plot(
            self,
            plot_obj,
            image_width,
            image_height,
            x_position=10
            ):
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
            h=image_height)
        self.pdf_object.set_y(y_position + image_height + 10)
        plot_obj.close()


    def _plot_fraction_of_empty_records(self, frac, data_analyst_no):
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
        plt.title('Counts_frac vs. Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Counts_frac')
        plt.grid(True)
        plt.xlim(0, 1)
        plt.savefig(os.path.join(
            self.output_path,
            f"Data Analysis {data_analyst_no}: Counts frac vs Threshold.png"
            ))
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer


    def _plot_distribution(
            self,
            data,
            plot_name,
            bins=10
            ):
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
        sns.displot(
            data,
            kde=True,
            bins=bins
            )
        plt.savefig(os.path.join(
            self.output_path,
            plot_name
            )
            )
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer


    def _add_analyst_title(self, data_analyst_no):
        """
        Adds the title of the analysis section to the PDF.

        Parameters
        ----------
        data_analyst_no : int
            Analysis identifier used in the title.
        """
        self._add_pdf_cell(f"Data Analysis: {data_analyst_no}")


    def _add_basic_statistics(self, data, data_analyst_no):
        """
        Adds basic statistics to the PDF, including the number of animals and
        fraction of empty records.

        Parameters
        ----------
        data : TrajectoriesFrame
            Input data for analysis.
        data_analyst_no : int
            Analysis identifier.
        """
        number_of_animals = len(data.get_users())
        self._add_pdf_cell(f"Number of animals: {number_of_animals}")

        frac = fraction_of_empty_records(data, resolution='1H')
        frac_median = frac.median()
        frac_mean = frac.mean()

        plot_obj = self._plot_fraction_of_empty_records(
            frac=frac,
            data_analyst_no=data_analyst_no
            )
        self._add_pdf_cell(
            f"Fraction of empty records (median): {frac_median}"
            )
        self._add_pdf_cell(
            f"Fraction of empty records (mean): {frac_mean}"
            )

        self._add_pdf_plot(plot_obj,100,60)


    def _add_total_records_statistics(self, data, data_analyst_no):
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

        self._add_pdf_cell(
            f"Total number of records (median): {count_median}"
            )
        self._add_pdf_cell(
            f"Total number of records (mean): {count_mean}",
            )

        plot_obj = self._plot_distribution(
            data=count,
            plot_name=f"Data Analysis {data_analyst_no}: "
                        f"Total numbers of records.png"
                        )

        self._add_pdf_plot(plot_obj,60,60)


    def _add_records_per_time_frame(self, data, data_analyst_no, resolution):
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
                    f"Count per {resolution}.png"
                        )

        self._add_pdf_plot(plot_obj,60,60)


    def _add_trajctories_duration(self, data, data_analyst_no, resolution):
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
            data,
            resolution=resolution,
            count_empty=False
            )

        self._add_pdf_cell(f"Trajectories duration ({resolution} median): "
                           f"{trajectories_duration_1H.median()}")

        plot_obj = self._plot_distribution(
            data=trajectories_duration_1H,
            plot_name=f"Data Analysis {data_analyst_no}: "
                    f"Trajectories duration ({resolution}) distribution.png"
                        )
        self._add_pdf_plot(plot_obj,60,60)


    def _add_no_of_consecutive_records(
            self,
            data,
            data_analyst_no,
            resolution
            ):
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

        self._add_pdf_cell(f"Median of consecutive records ({resolution}):"
                           f" {consecutive_1h_median}")

        plot_obj = self._plot_distribution(
            data=consecutive_1h,
            plot_name=f"Data Analysis {data_analyst_no}: "
                    f"Distribution of consecutive records ({resolution}).png"
                        )
        self._add_pdf_plot(plot_obj,60,60)


    def _add_average_temporal_resolution(self, data, data_analyst_no):
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
        temporal_df = data.reset_index(level=1).groupby(level=0).apply(
            lambda x: x.datetime - x.datetime.shift()
            )
        temporal_df_median = temporal_df.median()
        temp_res_animal = temporal_df.groupby(level=0).median()
        in_minutes = temporal_df[~temporal_df.isna()].dt.total_seconds()/60
        in_minutes_filtred = in_minutes[in_minutes > 0]

        self._add_pdf_cell(f"Median temporal resolution: "
                           f"{temporal_df_median}")

        self._add_pdf_cell(f"Max temporal resolution: "
                           f"{in_minutes_filtred.max()}")

        plot_obj = self._plot_distribution(
            data=in_minutes_filtred,
            plot_name=f"Data Analysis {data_analyst_no}: "
                    f"The distribution of average temporal resolution.png"
                        )
        self._add_pdf_plot(plot_obj,60,60)


class DataFilter:

    def __init__(self, pdf_object:FPDF) -> None:
        self.pdf_object = pdf_object
        self.allowed_minutes = [1, 5, 10, 15, 20, 30, 60]
        self.day_window = 21

    def _add_pdf_cell(self,txt_to_add:str):
        """
        Adds a single line of text to the PDF.

        Parameters
        ----------
        txt_to_add : str
            Text to be added to the PDF document.
        """
        self.pdf_object.cell(200, 5, txt=txt_to_add, ln=True, align='L')

    def _match_timedelta(self, timedelta):
        closest_minute = min(
            self.allowed_minutes,
            key=lambda x: abs(pd.Timedelta(minutes=x) - timedelta)
            )
        return f"{closest_minute}min", closest_minute


    def _select_time_window(self, closest_minute):
        per_hour = (60/closest_minute)
        per_day = per_hour * 24
        return int(self.day_window * per_day)


    def _extract_best_coverage(self, data, data_coverage):
        extracted = []
        for an_id, an_val in data_coverage.groupby(level=0):
            if an_val.isna().all():
                continue
            max_date = an_val.idxmax()[1]
            till_EOD = pd.to_datetime(
                max_date.date() + pd.Timedelta(days=1)
                ) - max_date
            max_date += till_EOD
            an_val = an_val.reset_index()
            min_date = max_date - pd.Timedelta(days=self.day_window)
            cur_df = data.loc[an_id].reset_index()
            selected = cur_df[
                (cur_df['datetime'] <= max_date) & (cur_df['datetime'] >= min_date)
                ]

            selected['user_id'] = an_id
            selected = selected.set_index(['user_id','datetime'])
            extracted.append(selected)

        return pd.concat(extracted)

    def _convert_to_unix(self, group):
        group['time'] = group['time'].apply(
            lambda x: int(pd.to_datetime(x).timestamp())
            )
        return group

    def select_best_period(self, data):
        temporal_df = data.reset_index(level=1).groupby(level=0).apply(
            lambda x: x.datetime - x.datetime.shift()
            )
        temporal_df_median = temporal_df.median()
        avg_temp_res_str, avg_temp_res_int = self._match_timedelta(temporal_df_median)
        resampled = data.groupby(level=0).resample(avg_temp_res_str, level=1).count().iloc[:,0]

        check_time = self._select_time_window(avg_temp_res_int)
        resampled[resampled > 1] = 1
        data_coverage = resampled.groupby(level=0).apply(
            lambda x: x.rolling(check_time).sum()/check_time
            )
        extracted = self._extract_best_coverage(data, data_coverage)

        self._add_pdf_cell("Selecting best periods - maximising data coverage")
        self._add_pdf_cell(f"Avg temporal resolution: {temporal_df_median}")
        self._add_pdf_cell(f"Resample value (str): {avg_temp_res_str}")
        self._add_pdf_cell(f"Resample value (int): {avg_temp_res_int}")
        self._add_pdf_cell(f"Selected time window for {self.day_window} "
                           f"days (timecheck): {check_time}")
        self.pdf_object.add_page()

        return TrajectoriesFrame(extracted)

    def filter_data(self, data):
        temporal_df = data.reset_index(level=1).groupby(level=0).apply(
            lambda x: x.datetime - x.datetime.shift()
            )
        temp_res_animal_ex = temporal_df.groupby(level=0).median()

        # FRACTION OF MISSING RECORDS < 0.6
        frac_filtr = fraction_of_empty_records(data, '1H')
        level1 = set(frac_filtr[frac_filtr < 0.6].index)

        # MORE THAN 20 DAYS OF DATA
        traj_dur_filtr = user_trajectories_duration(data, '1D')
        level2 = set(traj_dur_filtr[traj_dur_filtr > 20].index)

        level3 = set(temp_res_animal_ex[temp_res_animal_ex <= '30min'].index)

        # USER FILTRATION WITH ULOC METHOD
        selection_lvl2 = level1.intersection(level2)
        selection_lvl3 = selection_lvl2.intersection(level3)
        filtred_data = data.uloc(list(selection_lvl3))

        self._add_pdf_cell("Filtration with users statistics")

        self._add_pdf_cell(f"Number of animals on level 1: {len(level1)}")
        self._add_pdf_cell(f"Max fraction: {frac_filtr.max()}")
        self._add_pdf_cell(f"Min fraction: {frac_filtr.min()}")

        self._add_pdf_cell(f"Number of animals on level 2: {len(level2)}")
        self._add_pdf_cell(f"Max duration: {traj_dur_filtr.max()}")
        self._add_pdf_cell(f"Min duration: {traj_dur_filtr.min()}")

        self._add_pdf_cell(f"Number of animals on level 3: {len(level3)}")
        self._add_pdf_cell(f"Max temp res: {temp_res_animal_ex.max()}")
        self._add_pdf_cell(f"Min temp res: {temp_res_animal_ex.min()}")

        self._add_pdf_cell(f"TOTAL FILTRED ANIMALS: "
                           f"{len(filtred_data.get_users())}")

        self.pdf_object.add_page()

        return TrajectoriesFrame(filtred_data)

    def sort_data(self, data):
        data_sorted = data.sort_index(level=[0,1])
        data_sorted = data_sorted.to_crs(dest_crs = 4326, cur_crs = 3857)
        df1 = data_sorted.reset_index()
        data_prepared = df1.reset_index(drop=True)[['user_id', 'datetime', 'lon', 'lat']]
        data_prepared.columns = ['user_id','time', 'latitude', 'longitude']
        data_prepared = self._convert_to_unix(data_prepared)
        return data_prepared.groupby('user_id')


class LabelsCalc:

    def __init__(self, pdf_object:FPDF, output_path) -> None:
        self.pdf_object = pdf_object
        self.output_path = output_path

    def _add_pdf_cell(self,txt_to_add:str):
        """
        Adds a single line of text to the PDF.

        Parameters
        ----------
        txt_to_add : str
            Text to be added to the PDF document.
        """
        self.pdf_object.cell(200, 5, txt=txt_to_add, ln=True, align='L')

    def _add_pdf_plot(
            self,
            plot_obj,
            image_width,
            image_height,
            x_position=10
            ):
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
            h=image_height)
        self.pdf_object.set_y(y_position + image_height + 10)
        plot_obj.close()

    def _compute_intervals(self, labels, times, max_time_between=86400):
        """
        Compute stop and moves intervals from the list of labels.

        Parameters
        ----------
            labels: 1d np.array of integers
            times: 1d np.array of integers. `len(labels) == len(times)`.
        Returns
        -------
            intervals : array-like (shape=(N_intervals, 3))
                Columns are "label", "start_time", "end_time"
        """
        trajectory = np.hstack([labels.reshape(-1, 1), times.reshape(-1, 3)])
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

    def _process_user(self, user_data):
        user_id, group, rs1, rs2, min_staying_times = user_data
        dfs_list = []
        group = group.sort_values('time')
        data = group[['latitude', 'longitude', 'time']].values
        for r1 in rs1:
            for r2 in rs2:
                for min_staying_time in min_staying_times:
                    model = Infostop(
                        r1=r1,
                        r2=r2,
                        label_singleton=False,
                        min_staying_time=min_staying_time,
                        max_time_between=86400,
                        min_size=2
                    )
                    try:
                        labels = model.fit_predict(data)
                        trajectory = self._compute_intervals(labels, data)
                        trajectory = pd.DataFrame(trajectory, columns=['labels', 'lat', 'lon', 'start', 'end'])
                        trajectory = trajectory[trajectory.labels != -1]

                        total_stops = len(np.unique(labels))
                        results = {
                            'animal_id': user_id,
                            'Trajectory': trajectory,
                            'Total_stops': total_stops,
                            'R1': r1,
                            'R2': r2,
                            'Tmin': min_staying_time
                        }
                        dfs_list.append(results)
                    except Exception as e:
                        print(f"Error processing user {user_id}: {e}")
        return dfs_list

    def calc_params_matrix2(self, data):
        rs1 = np.logspace(1, 2, 20, base=50)
        rs2 = np.logspace(1, 2, 20, base=50)
        min_staying_times = np.logspace(np.log10(600), np.log10(7200), num=20)

        user_data_list = [(user_id, group, rs1, rs2, min_staying_times) for user_id, group in data]

        results = []
        # with ProcessPoolExecutor(max_workers=8) as executor:
        #     for dfs in tqdm(executor.map(self._process_user, user_data_list), total=len(user_data_list)):
        #         results.extend(dfs)

        gc.collect()
        results = Parallel(n_jobs=-1)(
            delayed(self._process_user)(user_data) for user_data in user_data_list
        )

        return pd.DataFrame(results)


    def calc_params_matrix(self, data):
        rs1 = np.logspace(1, 2, 20, base=50)
        rs2 = np.logspace(1, 2, 20, base=50)
        min_staying_times = np.logspace(np.log10(600), np.log10(7200), num=20)

        dfs_list = []
        for user_id, group in tqdm(data, total=len(data)):
                group = group.sort_values('time')
                data = group[['latitude', 'longitude','time']].values
                for r1 in rs1:
                    for r2 in rs2:
                        for min_staying_time in min_staying_times:
                            model = Infostop(r1=r1,
                                            r2=r2,
                                            label_singleton=False,
                                            min_staying_time=min_staying_time,
                                            max_time_between=86400,
                                            min_size=2)
                            try:
                                labels = model.fit_predict(data)
                                trajectory = self._compute_intervals(labels, data)
                                trajectory = pd.DataFrame(trajectory, columns=['labels', 'lat', 'lon', 'start', 'end'])
                                trajectory = trajectory[trajectory.labels != -1]

                                total_stops = len(np.unique(labels))
                                results = {
                                    'animal_id': user_id,
                                    'Trajectory': trajectory,
                                    'Total_stops': total_stops,
                                    'R1': r1,
                                    'R2': r2,
                                    'Tmin': min_staying_time
                                }
                                dfs_list.append(results)
                            except Exception as e:
                                print(f"Error processing user {user_id}: {e}")

        return pd.DataFrame(dfs_list)


    def _plot_param(self, param, stabilization_point_index, x, y, dy_dx):

        fig, ax1 = plt.subplots(figsize=(25, 20))
        ax1.plot(x, y, 'o', label='Original data', alpha=1, linestyle='-',linewidth=2)

        ax1.set_xlabel(f'{param}')
        ax1.set_ylabel('Total stops')

        ax1.axvline(stabilization_point_index, color='orange', linestyle='-', label=f'Stable point {param} = {stabilization_point_index:.0f}')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)

        ax2 = ax1.twinx()
        ax2.plot(x, dy_dx, label='First derivate', color='red', linewidth=2, linestyle='-')
        ax2.axhline(0, color='red', linestyle='--', linewidth=1, label='Zero line (dy/dx)')

        ax2.set_ylabel('First derivative')

        for a, b in zip(x, dy_dx):
            ax2.text(a, b, str(round(b, 5)), ha='center', va='bottom', color='red')

        ax1.legend(loc='lower right')
        ax2.legend(loc='upper right')

        buffer = BytesIO()
        plt.title(f"Median Total Stops by {param}")
        plt.savefig(os.path.join(
            self.output_path,
            f"Median Total Stops by {param}"
            )
        )
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer

    def _choose_param_value(self, data, param):

        if param == 'R1':
            data = data[(data['R1'] >= 50) & (data['R1'] <= 1000)]
        if param == 'Tmin':
            data = data[(data['Tmin'] >= 600) & (data['Tmin'] <= 3600)]

        result_med = data.groupby(['animal_id', param]).median()
        result_med_agg = result_med.groupby(level=param)['Total_stops'].median()


        x = result_med_agg.index
        y = result_med_agg.values

        dy_dx = np.diff(y)
        dy_dx = np.insert(dy_dx, 0, 0)

        dxdy = pd.DataFrame({'de_value': np.abs(dy_dx)}).sort_values(by='de_value')
        stabilization_x = x[dxdy.index[:5]]
        filtred = data[data[param].isin(stabilization_x)]
        suma_total_stops = filtred.groupby(param)['Total_stops'].sum()
        stabilization_point_index = suma_total_stops.idxmax()

        plot_obj = self._plot_param(param,stabilization_point_index,x,y,dy_dx)

        self._add_pdf_cell(f"{param} value: {stabilization_point_index}")
        self._add_pdf_plot(plot_obj,200,150)

        return stabilization_point_index


    def choose_best_params(self, sensitivity_matrix):
        r1 = self._choose_param_value(sensitivity_matrix, 'R1')
        r2 = self._choose_param_value(sensitivity_matrix, 'R2')
        self.pdf_object.add_page()
        Tmin = self._choose_param_value(sensitivity_matrix, 'Tmin')

        return r1, r2, Tmin

    def calc_labels(self, data, r1, r2, Tmin):
        results = []
        for user_id, group in tqdm(data, total=len(data)):
            group = group.sort_values('time')
            data = group[['latitude', 'longitude', 'time']].values
            model = Infostop(r1=r1,
                            r2=r2,
                            label_singleton=False,
                            min_staying_time=Tmin,
                            max_time_between=86400,
                            min_size=2)
            labels = model.fit_predict(data)
            trajectory = np.hstack([labels.reshape(-1, 1), data.reshape(-1, 3)])
            trajectory = pd.DataFrame(trajectory, columns=['labels', 'lat', 'lon', 'time'])
            trajectory = trajectory[trajectory.labels != -1]
            trajectory['animal_id'] = user_id

            results.append(pd.DataFrame(trajectory, columns=['animal_id', 'labels', 'lat', 'lon', 'time']))
        return pd.concat(results, ignore_index=True)

    def calculate_infostop(self, data):
        self._add_pdf_cell("Infostop calculations")
        sensitivity_matrix = self.calc_params_matrix(data)
        r1, r2, Tmin = self.choose_best_params(sensitivity_matrix)
        final_data = self.calc_labels(data, r1, r2, Tmin)
        self._add_pdf_cell(f"Final number of animals: {final_data['animal_id'].nunique()}")
        return final_data



class InfoStopData():

    def __init__(self, data, data_name, output_dir):
        self.clean_data = data
        self.animal_name = data_name
        self.output_dir = os.path.join(output_dir,data_name)
        self.pdf = FPDF()

        os.mkdir(self.output_dir)

        self.pdf.add_page()
        self.pdf.set_font("Arial", size=9)

        self.pdf.cell(
            200,
            10,
            txt = f"{self.animal_name}",
            ln=True,
            align='C'
            )


    def calculate_all(self, raports:bool=False, infostop_params_manual:bool=False, r1=None, r2=None, Tmin=None):

        raport = DataAnalystInfostop(self.pdf,self.output_dir)
        data_filter = DataFilter(self.pdf)
        labels_calculator = LabelsCalc(self.pdf,self.output_dir)

        raport.generate_raport(data = self.clean_data, data_analyst_no=1)
        extracted_data = data_filter.select_best_period(data=self.clean_data)
        raport.generate_raport(data = extracted_data, data_analyst_no=2)
        filtred_data = data_filter.filter_data(data=extracted_data)
        raport.generate_raport(data = filtred_data, data_analyst_no=3)
        sorted_data = data_filter.sort_data(filtred_data)
        trajectory_processed_data = labels_calculator.calculate_infostop(sorted_data)

        trajectory_processed_data.to_csv(os.path.join(self.output_dir, f'Trajectory_processed_{self.animal_name}.csv'))


        self.pdf.output(os.path.join(self.output_dir, f"{self.animal_name}.pdf"))
