import pandas as pd

from logger import Logger


class DataLoader:
    def __init__(self):
        self.logger = Logger()

    def read_raw_data(self, file_path: str):
        self.logger.info(f"Reading file: {file_path}")
        data = pd.read_csv(file_path, index_col=0)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data = data.set_index('animal_id')
        return data
