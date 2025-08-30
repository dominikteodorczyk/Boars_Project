import pandas as pd

from logger import Logger


class DataLoader:
    """
    Class to load and preprocess raw trajectory data from CSV files.
    """
    def __init__(self):
        """
        Initialize the DataLoader with a logger.
        """
        self.logger = Logger()

    def read_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Read raw trajectory data from a CSV file and preprocess it.

        Args:
            file_path (str): Path to the CSV file containing raw trajectory data.
        Returns:
            pd.DataFrame: Preprocessed trajectory data with 'animal_id' as index and 'time' as datetime.
        """
        self.logger.info(f"Reading file: {file_path}")
        data = pd.read_csv(file_path, index_col=0)
        data['time'] = pd.to_datetime(data['time'], unit='s')
        data = data.set_index('animal_id')
        return data
