from src.utils.dataIO import DataPrepocessing
from src.utils.istop import InfoStopData

def main():
    data = DataPrepocessing('')
    clean_data, data_name = data.infostop_data_prepare()
    infostop_object = InfoStopData(data=clean_data, data_name=data_name)
    infostop_object.calculate_all()

if __name__ == "__main__":
    main()
