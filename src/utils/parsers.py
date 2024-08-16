import pandas as pd
import os
import logging
from pandas.errors import ParserError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_data_detector(id: str) -> bool:
    """
    Detects if a given ID string is marked as test data.

    This function checks whether the provided ID string contains
    the word "test" or "Test", indicating that it is test data.

    Args:
        id (str): The ID string to be checked for test indicators.

    Returns:
        bool: True if the ID contains "test" or "Test", False otherwise.
    """
    id_strings = id.replace('_', ' ').split()
    if 'test' in id_strings or 'Ida' in id_strings or 'Wild' in id_strings:
        return True
    else:
        return False


def raw_data_parser(
        input_path:str,
        cols:list,
        output_path:str = None
) -> None:
    """
    Parses raw data from a CSV file, processes specified columns,
    and optionally saves the output to a specified path.

    Args:
        input_path (str): The file path to the input CSV file.
        cols (list): The list of column names to be selected from
            the input CSV file. The columns must be in the following
            order:
            - The first column is the timestamp.
            - The second column is the user ID.
            - The third column is the longitude.
            - The fourth column is the latitude.
        output_path (str, optional): The directory path to save
            the parsed CSV file. If not provided, the output file
            will be saved in the current directory.

    Returns:
        None

    Example:
        raw_data_parser(
            input_path='data/raw_data.csv',
            cols=['timestamp', 'user', 'longitude', 'latitude'],
            id_col='user',
            output_path='data/processed'
        )
    """
    try:
        file_name = os.path.basename(input_path)

        df = pd.read_csv(input_path)[cols]
        rows_no = df.shape[0]

        df[cols[1]] = df[cols[1]].astype(str)

        df = df[~df[cols[1]].apply(test_data_detector)]

        map_values = {
            val:num for val, num in zip(
                df[cols[1]].unique(), range(
                    len(df[cols[1]].unique())
                    )
                )
            }

        df[cols[1]] = df[cols[1]].map(map_values)

        try:
            df[cols[0]] = pd.to_datetime(df[cols[0]])
            df[cols[0]] = df[cols[0]].dt.strftime('%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError, ParserError) as e:
            logging.error(f'Time parser error: {e}')



        df.columns = ['datetime','user_id','lon', 'lat']
        df = df.reset_index(drop=True)

        if output_path:
            df.to_csv(
                os.path.join(
                    output_path,
                    f'parsed_{file_name}'
                    ),
                    index=False
                )
        else:
            df.to_csv(
                f'parsed_{file_name}',
                index=False
                )
        logging.info(f'{file_name} parsed successfully. Rows {rows_no}/{df.shape[0]}')
    except Exception as e:
        logging.error(f'Error in {file_name}: \n{e}')


def multi_raw_data_parser(
        data_dict:dict[str:list],
        output_path:str = None
) -> None:
    """
    Parses raw data from multiple CSV files based on a dictionary
    of file paths and column specifications,and optionally saves
    the outputs to a specified path.

    Args:
        data_dict (dict[str, list]): A dictionary where the keys are
            file paths to input CSV files and the valuesare lists of
            column names to be selected from each input CSV file. The
            columns in each list must be in the following order:
            - The first column is the timestamp.
            - The second column is the user ID.
            - The third column is the longitude.
            - The fourth column is the latitude.
        output_path (str, optional): The directory path to save the
            parsed CSV files.If not provided, the output files will
            be saved in the current directory.

    Returns:
        None

    Example:
        data_dict = {
            'data/raw_data1.csv': [
                'timestamp',
                'user',
                'longitude',
                'latitude'],
            'data/raw_data2.csv': [
                'time',
                'id',
                'lon',
                'lat']
        }
        dir_raw_data_parser(data_dict, output_path='data/processed')
    """
    for key, value in data_dict.items():
        raw_data_parser(
            input_path=key,
            cols=value,
            output_path=output_path
            )
    logging.info(f'Parsing done.')



