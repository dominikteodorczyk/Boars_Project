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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

class DataAnalystInfostop:
    """
    Initializes the DataAnalysisReport with a PDF instance.

    Args:
        pdf_object (FPDF): PDF instance where the analysis
            results will be written.
    """
    def __init__(self, pdf_object) -> None:
        self.pdf_object = pdf_object

    def generate_raport(self, data, data_analyst_no: int):
        """
        Generates a data analysis report, adding results to the PDF.

        Args:
            data (TrajectoriesFrame): Input data for analysis.
            data_analyst_no (int): Analysis number, used in titles and file names.

        Description:
            Runs a complete data analysis pipeline, sequentially generating statistics
            and plots and adding them to the PDF instance.
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


    def _add_pdf_cell(self,txt_to_add:str):
        self.pdf_object.cell(200, 5, txt=txt_to_add, ln=True, align='L')

    def _add_pdf_plot(
            self,
            plot_obj,
            image_width,
            image_height,
            x_position=10
            ):

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
        plt.savefig(
            f"Data Analysis {data_analyst_no}: Counts frac vs Threshold.png"
            )
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

        buffer = BytesIO()
        sns.displot(
            data,
            kde=True,
            bins=bins
            )
        plt.savefig(plot_name)
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        return buffer


    def _add_analyst_title(self, data_analyst_no):
        self._add_pdf_cell(f"Data Analysis: {data_analyst_no}")


    def _add_basic_statistics(self, data, data_analyst_no):
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


    def _add_no_of_consecutive_records(self, data, data_analyst_no, resolution):

        consecutive_1h = consecutive_record(data, resolution=resolution)
        consecutive_1h_median = consecutive_1h.median()

        self._add_pdf_cell(f"The median of consecutive records ({resolution}): "
                           f"{consecutive_1h_median}")

        plot_obj = self._plot_distribution(
            data=consecutive_1h,
            plot_name=f"Data Analysis {data_analyst_no}: "
                    f"The distribution of consecutive records ({resolution}).png"
                        )
        self._add_pdf_plot(plot_obj,60,60)


    def _add_average_temporal_resolution(self, data, data_analyst_no):
        # temporal_df = data.reset_index(level=1)
        temporal_df = data.reset_index(level=1).groupby(level=0).apply(lambda x: x.datetime - x.datetime.shift())
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



class InfoStopData():

    def __init__(self, data, data_name):
        self.clean_data = data
        self.animal_name = data_name
        self.pdf = FPDF()

        self.pdf.add_page()
        self.pdf.set_font("Arial", size=9)

        self.pdf.cell(
            200,
            10,
            txt=f"{self.animal_name}",
            ln=True,
            align='C'
            )

    def data_analyst(self, data, data_analyst_no):
        raport = DataAnalystInfostop(self.pdf)
        raport.generate_raport(data, data_analyst_no)


#     def selecting_best_periods(self):
#         pass


#     def filtration(self):
#         pass


#     def sort_by_timestamp(self, data) -> pd.DataFrame:
#         # get data from extraction 2 step and sort them
#         # create a grouped df for sensitivity analyst and infostop calculations
#         pass


#     def infostop_calculation(self, data, r1, r2, Tmin) -> pd.DataFrame:
#         # create df of data with labels
#         return None


    def calculate_all(self, raports:bool=False, infostop_params_manual:bool=False, r1=None, r2=None, Tmin=None):

        self.data_analyst(data = self.clean_data, data_analyst_no=1)
#         # self.selecting_best_periods()
#         # self.data_analyst()
#         # self.filtration()
#         # self.data_analyst()
#         # self.sort_by_timestamp
#         # infostop_params = SensitiveAnalyst()
#         # self.infostop_calculation()
#         # self.write_data()
#         # self.write_stats()
        self.pdf.output(f"{self.animal_name}.pdf")

#     def calculate_sensitivity(self):

#         self.data_analyst()
#         self.selecting_best_periods()
#         self.data_analyst()
#         self.filtration()
#         self.data_analyst()
#         self.sort_by_timestamp
#         infostop_params = SensitiveAnalyst()



#     def calculate_infostop_with_params(self, raports:bool=False, r1=None, r2=None, Tmin=None):

#         self.data_analyst()
#         self.selecting_best_periods()
#         self.data_analyst()
#         self.filtration()
#         self.data_analyst()
#         self.sort_by_timestamp
#         self.infostop_calculation()
#         self.write_data()
#         self.write_stats()




# class SensitiveAnalyst():

#     def __init__(self):
#         pass
