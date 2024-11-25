from src.utils.dataIO import DataPrepocessing
from src.utils.istop import InfoStopData
import os

PARSED_DATA_DIR = ''

def main():

    file_paths = [
        os.path.join(PARSED_DATA_DIR, file)
        for file in os.listdir(PARSED_DATA_DIR)
        if os.path.isfile(os.path.join(PARSED_DATA_DIR, file))
        ]
    os.mkdir(os.path.join(
                        os.path.dirname(__file__),
                        'infostop_output'
                        ))

    for parsed_file in file_paths:
        try:
            data = DataPrepocessing(parsed_file)
            clean_data, data_name = data.infostop_data_prepare()



            infostop_object = InfoStopData(
                data=clean_data,
                data_name=data_name,
                output_dir = os.path.join(
                        os.path.dirname(__file__),
                        'infostop_output'
                        )
            )
            infostop_object.calculate_all()
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main()
