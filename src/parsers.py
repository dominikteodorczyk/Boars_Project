import json
import pandas as pd
import os
import logging
from typing import Optional
from pandas.errors import ParserError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_data_detector(id: str) -> bool:
    """
    Detects if a given ID string is marked as test data.

    This function checks whether the provided ID string contains
    certain keywords such as 'test', 'Ida', or 'Wild', which are
    indicators that the data might be test data.

    Args:
        id (str): The ID string to check for test indicators.

    Returns:
        bool: True if the ID contains 'test', 'Ida',
            or 'Wild', False otherwise.
    """
    id_strings = id.replace('_', ' ').split()
    if 'test' in id_strings or 'Ida' in id_strings or 'Wild' in id_strings:
        return True
    else:
        return False


def parse_id(dataframe:pd.DataFrame, cols:list) -> Optional[pd.DataFrame]:
    """
    Processes the user ID column by removing test data and mapping
    unique IDs to integers.

    This function converts the specified user ID column to string
    format, removes rows where the ID is detected as test data,
    and then maps the remaining unique IDs to integers for easier
    processing.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the data.
        cols (list): List of column names where the second column is
            the user ID.

    Returns:
        pd.DataFrame: The processed DataFrame with test data removed and
            user IDs mapped to integers.
    """
    try:
        dataframe[cols[1]] = dataframe[cols[1]].astype(str)
        dataframe = dataframe[~dataframe[cols[1]].apply(test_data_detector)]
        dataframe[cols[1]] = dataframe[cols[1]].map(
            {val: num for num, val in enumerate(dataframe[cols[1]].unique())}
            )
        return dataframe
    except Exception as e:
        logging.error(f'ID parser error: {e}')


def parse_time(dataframe:pd.DataFrame, cols:list) -> Optional[pd.DataFrame]:
    """
    Converts the time column to a standard datetime format.

    This function processes the specified time column, converting it to a
    standard datetime format ('%Y-%m-%d %H:%M:%S') for consistency.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing
            the data.
        cols (list): List of column names where the first column
            is the timestamp.

    Returns:
        pd.DataFrame: The processed DataFrame with the time column
            converted to a standard format.
    """
    try:
        dataframe[cols[0]] = pd.to_datetime(dataframe[cols[0]])
        dataframe[cols[0]] = dataframe[cols[0]].dt.strftime('%Y-%m-%d %H:%M:%S')
        return dataframe
    except (ValueError, TypeError, ParserError) as e:
        logging.error(f'Time parser error: {e}')


def data_structuring(dataframe:pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Structures the DataFrame by renaming columns and resetting
    the index.

    This function renames the columns of the DataFrame to
    'datetime', 'user_id', 'lon', and 'lat', and then resets
    the index for a clean output.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The structured DataFrame with renamed
            columns and reset index.
    """
    try:
        dataframe.columns = ['datetime','user_id','lon', 'lat']
        dataframe.groupby('user_id', group_keys=False).apply(
            lambda x: x.sort_values('datetime')
            )
        return dataframe.reset_index(drop=True).dropna()
    except Exception as e:
        logging.error(f'Data structuring error: {e}')


def filter_by_month_range(data, start, end, in_out=True):
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
    months = data['datetime'].dt.month

    if start <= end:
        condition = months.between(start, end)
    else:
        condition = (months >= start) | (months <= end)

    if in_out:
        result = data[condition]
    else:
        result = data[~condition]
    return result


def data_write(dataframe:pd.DataFrame, filename:str, output_path:Optional[str]=None):
    """
    Writes the processed DataFrame to a CSV file.

    This function saves the processed DataFrame to a CSV file
    with a name prefixed by 'parsed_'. If an output path is specified,
    it saves the file in that directory.

    Args:
        dataframe (pd.DataFrame): The processed DataFrame to be saved.
        filename (str): The original filename of the data.
        output_path (str, optional): The directory to save the output file.
            If None, the file will be saved in the current directory.
    """
    try:
        if output_path:
            dataframe.to_csv(
                os.path.join(
                    output_path,
                    f'parsed_{filename}'
                    ),
                    index=False
                )
        else:
            dataframe.to_csv(
                f'parsed_{filename}',
                index=False
                )
    except Exception as e:
        logging.error(f'Writing error: {e}')


def raw_data_parser(
        input_path:str,
        cols:list,
        output_path:Optional[str] = None,
        breeding_periods:Optional[list] = None
) -> None:
    """
    Parses and processes raw data from a CSV file.

    This function reads a CSV file, processes the specified
    columns (e.g., timestamp, user ID, longitude, and latitude),
    and saves the processed data to a new CSV file.

    Args:
        input_path (str): The file path to the input CSV file.
        cols (list): List of column names to process, in the
            following order:
                     - The first column is the timestamp.
                     - The second column is the user ID.
                     - The third column is the longitude.
                     - The fourth column is the latitude.
        output_path (str, optional): The directory to save the
            processed CSV file. If None, saves in the current directory.

    Returns:
        None
    """
    try:
        file_name = os.path.basename(input_path)

        raw_data = pd.read_csv(input_path, low_memory=False)[cols]
        rows_no = raw_data.shape[0]

        id_parsed = parse_id(dataframe=raw_data, cols=cols)
        time_parsed = parse_time(dataframe=id_parsed, cols=cols)
        data_structured = data_structuring(time_parsed)
        if not breeding_periods and data_structured.shape[0] != 0:
            data_write(
                dataframe=data_structured,
                filename=file_name,
                output_path=output_path
                )
        elif len(breeding_periods) == 2 and data_structured.shape[0] != 0:
            in_breeding_periods = filter_by_month_range(
                data_structured,
                breeding_periods[0],
                breeding_periods[1],
                True
            )
            out_breeding_periods = filter_by_month_range(
                data_structured,
                breeding_periods[0],
                breeding_periods[1],
                False
            )
            if in_breeding_periods.shape[0] != 0:
                data_write(
                    dataframe=in_breeding_periods,
                    filename=f'in_breeding_{file_name}',
                    output_path=output_path
                    )
            if out_breeding_periods.shape[0] != 0:
                data_write(
                    dataframe=out_breeding_periods,
                    filename=f'out_breeding_{file_name}',
                    output_path=output_path
                    )

        logging.info(
            f'{file_name} parsed successfully. '
            f'Changing the no. of rows: {rows_no}->{data_structured.shape[0]}'
            )

    except Exception as e:
        logging.error(f'Error in {file_name}: \n{e}')


def multi_raw_data_parser(
        data_dict:dict,
        periods_dict:dict,
        output_path:Optional[str] = None
) -> None:
    """
    Parses raw data from multiple CSV files.

    This function processes multiple CSV files based on
    the provided dictionary of file paths and corresponding column
    specifications, saving the results to a specified output path.

    Args:
        data_dict (dict[str, list]): A dictionary where keys are
            file paths to input CSV files,and values are lists of column
            names to be processed in the same order:
                - The first column is the timestamp.
                - The second column is the user ID.
                - The third column is the longitude.
                - The fourth column is the latitude.
        output_path (str, optional): The directory to save the processed
            files. If None, saves in the current directory.
    """
    for key, value in data_dict.items():
        breeding_periods = periods_dict[os.path.basename(key)]
        raw_data_parser(
            input_path=key,
            cols=value,
            output_path=output_path,
            breeding_periods = breeding_periods
            )
    logging.info(f'Multiparsing done.')


def parse_data(
        json_source: Optional[str] = None,
        periods: Optional[str] = None,
        path: Optional[str] = None,
        cols: Optional[list[str]] = None,
        output_path: Optional[str] = None
    ) -> None:
    """
    Parses raw data from either a JSON configuration or a single file.

    This function supports two modes of operation:
    1. Using a JSON configuration file to process multiple CSV files.
    2. Parsing a single CSV file with specified columns.

    Args:
        json_source (Optional[str]): Path to a JSON file containing
            a mapping of file paths and column specifications. Required
            for processing multiple files.
        periods (Optional[str]): Path to a JSON file containing additional
            configuration (e.g., periods) for multi-file processing.
        path (Optional[str]): Path to a single CSV file for direct processing.
        cols (Optional[List[str]]): List of column names to be used
            when processing a single CSV file. Required if `path` is provided.
        output_path (Optional[str]): Directory to save processed file(s).
            Defaults to the current directory if not specified.

    Returns:
        None

    Raises:
        ValueError: If required arguments are missing or invalid.
        Exception: For any unexpected errors during processing.

    Examples:
        # Example with JSON configuration for multiple files:
        parse_data(
            json_source='data/config.json',
            periods='data/periods.json',
            output_path='data/processed'
        )

        # Example with a single CSV file:
        parse_data(
            path='data/raw_data.csv',
            cols=['timestamp', 'user', 'longitude', 'latitude'],
            output_path='data/processed'
        )
    """
    if json_source and periods:
        data_dict = json.load(open(json_source))
        periods_dict = json.load(open(periods))

        multi_raw_data_parser(
            data_dict=data_dict,
            periods_dict=periods_dict,
            output_path=output_path
            )
    elif not json_source:
        if type(path) == str and type(cols) == list:
            try:
                #TODO: add breeding periods options to single csv parsing
                raw_data_parser(
                    input_path=path,
                    cols=cols,
                    output_path=output_path
                    )
            except Exception as e:
                logging.error(f'Error in {path}: \n{e}')
    else:
        logging.error(f'Cant parse this data')



