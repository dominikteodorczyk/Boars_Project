import pandas as pd
import os

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

    file_name = os.path.basename(input_path)

    df = pd.read_csv(input_path)[cols]
    df[cols[1]] = df[cols[1]].astype(str)

    map_values = {
        val:num for val, num in zip(
            df[cols[1]].unique(), range(
                len(df[cols[1]].unique())
                )
            )
        }

    df[cols[1]] = df[cols[1]].map(map_values)
    df.timestamp = pd.to_datetime(df.timestamp)
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



