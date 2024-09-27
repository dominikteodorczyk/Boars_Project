import os
import glob
import logging
import pandas as pd
import numpy as np
from src.utils.dataIO import DataPrepocessing
from src.measures.measures import Measures



DATA_AFTER_INFOSTOP = '/home/dteodorczyk/Desktop/boars_repo/Boars_Project/test_data/trajctory_processed'


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main(path):

    parent_dir = os.path.abspath(path)
    animals_data = pd.DataFrame(columns=[
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
        "travel_times",
        "travel_times_params",
        "rog",
        "rog_params",
        "rog_over_time",
        "rog_over_time_params",
        "msd_distribution",
        "msd_distribution_params",
        "msd_curve",
        "msd_curve_params",
        "return_time_distribution",
        "return_time_distribution_params",
        "exploration_time",
        "exploration_time_params",
        ]
        )

    for csv_file in glob.glob(os.path.join(parent_dir, '*.csv')):

        try:
            logging.info(f"Processing file: {csv_file}")
            animal_data = {}

            data_preproccessing = DataPrepocessing(csv_file)
            data_preproccessing.scaling_laws_prepare()
            data = data_preproccessing.filter_data()

            min_label_no = [[value] for index, value in data_preproccessing.statistics.min_labels_no_after_filtration.items()][0][0]
            min_records_no = [[value] for index, value in data_preproccessing.statistics.min_records_no_before_filtration.items()][0][0]


            measures = Measures(data, calc_type='auto', min_label_no = min_label_no, min_records_no = min_records_no)
            measures.calculate_all()
            instance_attributes = [
                attr for attr in measures.stats.__dict__.keys() if not callable(
                    getattr(measures.stats, attr)
                    )
                ]

            animal_data = {
                'animal' : data_preproccessing.animal.rsplit('\\', 1)[-1],
                'animal_no':data_preproccessing.statistics.raw_animals_no,
                'animal_after_filtration':data_preproccessing.statistics.filtered_animals_no,
                'time_period':data_preproccessing.statistics.filtered_period,
                'min_label_no':[[f'user_id : {index}', f'no: {value}'] for index, value in data_preproccessing.statistics.min_labels_no_after_filtration.items()],
                'min_records':[[f'user_id : {index}', f'no: {value}'] for index, value in data_preproccessing.statistics.min_records_no_before_filtration.items()],
                'avg_duration':data_preproccessing.statistics.avg_duration,
                'min_duration':data_preproccessing.statistics.min_duration,
                'max_duration':data_preproccessing.statistics.max_duration,
                'overall_set_area':data_preproccessing.statistics.overall_set_area,
                'average_set_area':data_preproccessing.statistics.average_set_area,
                'min_area':data_preproccessing.statistics.min_area,
                'max_area':data_preproccessing.statistics.max_area
            }


            for statistic in instance_attributes:

                statistic_values = getattr(measures.stats, statistic)
                animal_data[f'{statistic}'] = statistic_values[1]

                if type(statistic_values[2]) == np.ndarray:
                    animal_data[f'{statistic}_params'] = tuple(statistic_values[2])
                else:
                    animal_data[f'{statistic}_params'] = statistic_values[2]

            animals_data = animals_data._append(animal_data,ignore_index = True)

        except Exception as e:
            logging.error(f"Error durring calculation {csv_file}: {e}")
            animals_data = animals_data._append({
                'animal' : data_preproccessing.animal.rsplit('\\', 1)[-1],
                'animal_no':data_preproccessing.statistics.raw_animals_no,
                'animal_after_filtration':data_preproccessing.statistics.filtered_animals_no,
                'time_period':data_preproccessing.statistics.filtered_period,
                'min_label_no':[[f'user_id : {index}', f'no: {value}'] for index, value in data_preproccessing.statistics.min_labels_no_after_filtration.items()],
                'min_records':[[f'user_id : {index}', f'no: {value}'] for index, value in data_preproccessing.statistics.min_records_no_before_filtration.items()],
                'avg_duration':data_preproccessing.statistics.avg_duration,
                'min_duration':data_preproccessing.statistics.min_duration,
                'max_duration':data_preproccessing.statistics.max_duration,
                'overall_set_area':data_preproccessing.statistics.overall_set_area,
                'average_set_area':data_preproccessing.statistics.average_set_area,
                'min_area':data_preproccessing.statistics.min_area,
                'max_area':data_preproccessing.statistics.max_area
            },ignore_index = True)


    animals_data.to_csv('animal_stats.csv')


if __name__ == "__main__":

    main(DATA_AFTER_INFOSTOP)
