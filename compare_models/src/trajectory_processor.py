import numpy as np
import pandas as pd


class TrajectoryProcessor:
    @staticmethod
    def filter_by_quartile(data: pd.DataFrame, quartile: float):
        unique_labels = data.groupby('animal_id')['labels'].nunique()
        quartile_value = unique_labels.quantile(quartile)
        return data[data.index.isin(unique_labels[unique_labels > quartile_value].index)]

    @staticmethod
    def resample_time(data: pd.DataFrame, freq: str, label_col: str = 'labels'):
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
    def compute_stats(df: pd.DataFrame):
        stats = df.groupby('animal_id').apply(lambda g: g.index.get_level_values('time').nunique())
        return stats.to_frame(name='time_count').assign(labels=df.groupby('animal_id')['labels'].nunique())
