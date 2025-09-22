import numpy as np
import pandas as pd


class TrajectoryProcessor:
    """
    Class to process trajectory data. Includes methods for filtering by quartile, resampling time intervals,
    and computing statistics on the trajectory data.

    Methods:
    - filter_by_quartile: Filters the data to include only animals with a number of unique labels above a specified quartile.
    - resample_time: Resamples the trajectory data to a specified frequency, assigning the most visited label in each interval.
    - compute_stats: Computes statistics such as the number of unique time points and labels for each animal.
    """
    @staticmethod
    def filter_by_quartile(data: pd.DataFrame, quartile: float) -> pd.DataFrame:
        """
        Filter the data to include only animals with a number of unique labels above the specified quartile.
        Will remove animals with very few unique labels. E.g., if quartile=0.25, will remove the 25% of animals with the fewest unique labels.

        Args:
            data (pd.DataFrame): DataFrame containing trajectory data with 'animal_id' and 'labels' columns.
            quartile (float): Quartile threshold (between 0 and 1) for filtering animals based on unique labels.
        Returns:
            pd.DataFrame: Filtered DataFrame containing only animals above the specified quartile of unique labels.
        """
        unique_labels = data.groupby('animal_id')['labels'].nunique()
        quartile_value = unique_labels.quantile(quartile)
        return data[data.index.isin(unique_labels[unique_labels > quartile_value].index)]

    @staticmethod
    def resample_time(data: pd.DataFrame, freq: str, label_col: str = 'labels') -> pd.DataFrame:
        """
        Resample the trajectory data to a specified frequency, assigning the most visited label in each interval.
        The most visited label is determined by the total duration spent at each label within the interval.

        Args:
            data (pd.DataFrame): DataFrame containing trajectory data with a datetime index and 'animal_id', 'labels', 'lat', 'lon' columns.
            freq (str): Resampling frequency (e.g., '1H' for 1 hour).
            label_col (str): Column name for labels. Default is 'labels'.
        Returns:
            pd.DataFrame: Resampled DataFrame with the most visited label for each time interval.
        """
        to_concat = {}
        for uid, group in data.groupby(level=0):
            group = group[~group['time'].duplicated()]
            group.set_index('time', inplace=True)
            group['duration'] = (group.index.to_series().shift(-1) - group.index).dt.total_seconds()
            group['duration'] = group['duration'].fillna(3600)

            def longest_visited_label(groupa):
                return groupa.groupby(label_col)['duration'].sum().idxmax() if not groupa.empty else np.nan

            group_resampled = group.resample(freq).apply(longest_visited_label)
            group_resampled = group_resampled.to_frame(name=label_col)

            unique_labels = group[[label_col, 'lat', 'lon']].drop_duplicates(subset=[label_col]).set_index(label_col)
            group_resampled = group_resampled.join(unique_labels, on=label_col).ffill().bfill()
            to_concat[uid] = group_resampled
        return pd.concat(to_concat)

    @staticmethod
    def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics such as the number of unique time points and labels for each animal.

        Args:
            df (pd.DataFrame): DataFrame containing trajectory data with 'animal_id', 'time', and 'labels' columns.
        Returns:
            pd.DataFrame: DataFrame with statistics for each animal, including 'time_count' and 'labels'.
        """
        stats = df.groupby('animal_id').apply(lambda g: g.index.get_level_values('time').nunique())
        return stats.to_frame(name='time_count').assign(labels=df.groupby('animal_id')['labels'].nunique())
