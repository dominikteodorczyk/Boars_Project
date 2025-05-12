"""
Module for calculating coefficients of association (Ca) between
moving agents based on spatiotemporal proximity.

Includes:
- `EventCa`: Calculates event-based Ca based on shared proximity
    in space and time.
- `TimeCa`: Calculates time-based Ca based on duration of shared
    proximity.
"""

from datetime import timedelta
from itertools import combinations, permutations
import time
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn import preprocessing
from intervaltree import IntervalTree
from concurrent.futures import ProcessPoolExecutor, as_completed

class CoeffAssociation:
    """
    Base class for association coefficient computation between
    individuals. Handles input formatting and user extraction.
    """
    def __init__(self) -> None:
        self.data = pd.DataFrame()
        pass

    def _get_users(self) -> list:
        """
        Extracts unique user IDs from the dataset.

        Returns:
            list: List of unique user IDs.
        """
        users = self.data.user_id.unique()
        return list(users)

    def input_data(
        self,
        data: pd.DataFrame,
        id_col: str = "user_id",
        timestamp: str = "datetime",
        lat: str = "lat",
        lon: str = "lon",
    ) -> None:
        """
        Loads and standardizes input tracking data.

        Args:
            data (pd.DataFrame): Input dataframe with tracking data.
            id_col (str): Name of the column with individual/user IDs.
            timestamp (str): Name of the timestamp column.
            lat (str): Name of the latitude column.
            lon (str): Name of the longitude column.
        """
        self.data = data.rename(
            columns={
                id_col: "user_id",
                timestamp: "datetime",
                lat: "lat",
                lon: "lon"
            }
        )
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])


class EventCa(CoeffAssociation):
    """
    Calculates event-based Coefficient of Association (Ca).
    Based on the number of meetings between individuals that fall
    within both a temporal and spatial threshold.

    Example:
    >>> ca = EventCa()
    >>> ca.input_data(data)
    >>> result = ca.compute(temporal=3600, distance=1000)
    """

    def __init__(self) -> None:
        super().__init__()

    def _calc_meetings_values(
        self, main_agent, secound_agent, td: timedelta, dd: int
    ) -> tuple:
        """
        Calculates number of events (meetings) between two individuals
        using iteration.

        Args:
            main_agent: ID of the main individual.
            secound_agent: ID of the secondary individual.
            td (timedelta): Maximum time difference to consider a meeting.
            dd (int): Maximum spatial distance (in meters).

        Returns:
            tuple: (Number of meetings, Missed events)
        """
        m_agent_data = (
            self.data[self.data["user_id"] == main_agent]
            .reset_index(drop=True)
            .sort_values("datetime")
        )
        s_agent_data = (
            self.data[self.data["user_id"] == secound_agent]
            .reset_index(drop=True)
            .sort_values("datetime")
        )

        am_value = 0
        m_value = 0

        for _, m_row in m_agent_data.iterrows():
            temp_dataframe = s_agent_data.copy()

            temp_dataframe["m_datetime"] = m_row["datetime"]
            temp_dataframe["m_lat"] = m_row["lat"]
            temp_dataframe["m_lon"] = m_row["lon"]
            temp_dataframe["time_diff"] = abs(
                temp_dataframe["datetime"] - temp_dataframe["m_datetime"]
            )

            filtred_dataframe = temp_dataframe[
                temp_dataframe["time_diff"] <= td
            ]
            if filtred_dataframe.shape[0] > 0:
                filtred_dataframe = filtred_dataframe.copy()
                filtred_dataframe.loc[:, "distance"] = filtred_dataframe.apply(
                    lambda x: geodesic(
                        (x["lat"], x["lon"]), (x["m_lat"], x["m_lon"])
                    ).meters,
                    axis=1,
                )
                final_dataframe = filtred_dataframe[
                    filtred_dataframe["distance"] <= dd
                ]
                if final_dataframe.shape[0] > 0:
                    am_value += 1
            else:
                m_value += 1

        return am_value, m_value

    def _calc_meetings_values_vector(
        self, main_agent, secound_agent, td: int, dd: int
    ) -> tuple:
        """
        Optimized vectorized version for calculating meetings.

        Args:
            main_agent: ID of the main individual.
            secound_agent: ID of the secondary individual.
            td (int): Time threshold in seconds.
            dd (int): Distance threshold in meters.

        Returns:
            tuple: (Number of meetings, Missed events)
        """
        m_agent_data = (
            self.data[self.data["user_id"] == main_agent]
            .reset_index(drop=True)
            .sort_values("datetime")
        )
        s_agent_data = (
            self.data[self.data["user_id"] == secound_agent]
            .reset_index(drop=True)
            .sort_values("datetime")
        )
        m_agent_data["datetime"] = m_agent_data["datetime"].astype("int64") // 10**9
        s_agent_data["datetime"] = s_agent_data["datetime"].astype("int64") // 10**9
        m_agent_data = m_agent_data.to_numpy()
        s_agent_data = s_agent_data.to_numpy()

        am_value = 0
        m_value = 0
        for m_row in m_agent_data:
            for s_row in s_agent_data:
                if abs(m_row[0] - s_row[0]) <= td:
                    if (
                        geodesic(
                            (s_row[3], s_row[2]), (m_row[3], m_row[2])
                        ).meters
                        <= dd
                    ):
                        am_value += 1
                        break

        m_value = m_agent_data.shape[0] - am_value
        return am_value, m_value

    def _compute_pair_ca(
            self, main_agent, secound_agent, td: int, dd: int
        ) -> tuple:
        """
        Computes the association coefficient between two individuals.

        Args:
            main_agent: ID of the first individual.
            secound_agent: ID of the second individual.
            td (int): Time threshold in seconds.
            dd (int): Distance threshold in meters.

        Returns:
            tuple: (Ca value, AB1 count, AB2 count, A-only, B-only)
        """
        time_threshold = timedelta(seconds=td)

        ab1, a_only = self._calc_meetings_values_vector(
            main_agent, secound_agent, td, dd
        )
        ab2, b_only = self._calc_meetings_values_vector(
            secound_agent, main_agent, td, dd
        )
        denominator = a_only + b_only
        if denominator > 0:
            return (2 * ab2) / denominator, ab1, ab2, a_only, b_only
        elif denominator == 0:
            return 1.0, ab1, ab2, a_only, b_only
        else:
            return 0, ab1, ab2, a_only, b_only

    def compute(
            self, temporal: int = 3600, distance: int = 1000
        ) -> pd.DataFrame:
        """
        Computes pairwise event-based Ca for all unique user pairs.

        Args:
            temporal (int): Time threshold in seconds (default: 3600).
            distance (int): Distance threshold in meters (default: 1000).

        Returns:
            pd.DataFrame: Result dataframe with
                columns [A, B, Ca, AB1, AB2, A-only, B-only]
        """
        users = self._get_users()
        pairs = list(combinations(users, 2))
        results_frame = pd.DataFrame(
            columns=["A", "B", "Ca", "AB1", "AB2", "A", "B"]
        )
        for users in pairs:
            ca, ab1, ab2, a_only, b_only = self._compute_pair_ca(
                users[0], users[1], temporal, distance
            )
            results_frame = pd.concat(
                [
                    results_frame,
                    pd.DataFrame(
                        [[users[0], users[1], ca, ab1, ab2, a_only, b_only]],
                        columns=["A", "B", "Ca", "AB1", "AB2", "A", "B"],
                    ),
                ],
                ignore_index=True,
            )
        return results_frame


class TimeCa(CoeffAssociation):
    """
    Calculates time-based Coefficient of Association (Ca).
    Based on overlapping durations where individuals are close in space.

    Example:
    >>> ca = TimeCa()
    >>> ca.input_data(data)
    >>> result = ca.compute(distance=1000)
    """
    def __init__(self) -> None:
        super().__init__()

    def _calc_meetings_values(
            self, main_agent, secound_agent
        ) -> float:
        """
        Computes the duration of co-location between two individuals.

        Args:
            main_agent: ID of the first individual.
            secound_agent: ID of the second individual.
            dd (int): Distance threshold in meters.

        Returns:
            tuple: (Total observation time, Time together, Ca value)
        """
        m_agent_data = self.data[
            self.data["user_id"] == main_agent
        ].sort_values("datetime")
        m_agent_data["start"] = m_agent_data["datetime"]
        m_agent_data["end"] = m_agent_data["datetime"].shift(-1)

        s_agent_data = self.data[
            self.data["user_id"] == secound_agent
        ].sort_values("datetime")
        s_agent_data["start"] = s_agent_data["datetime"]
        s_agent_data["end"] = s_agent_data["datetime"].shift(-1)

        distances = []
        time_together = []
        for _, row in m_agent_data.iterrows():
            df_time_sort = s_agent_data[
                (s_agent_data["start"] < row["end"])
                & (s_agent_data["end"] > row["start"])
            ]

            if df_time_sort.shape[0] != 0:
                for _sort, row_sort in df_time_sort.iterrows():
                    distance = geodesic(
                        (row_sort["lat"], row_sort["lon"]),
                        (row["lat"], row["lon"])
                    ).meters

                    common_start = max(row["start"], row_sort["start"])
                    common_end = min(row["end"], row_sort["end"])
                    period = common_end - common_start

                    distances.append(distance)
                    time_together.append(period)

        # max_time = (m_agent_data["end"].max() - m_agent_data["start"].min()).total_seconds()
        ca = 0
        if distances and time_together:
            dmax = max(distances)
            tsum = np.sum(time_together)
            for i in range(len(time_together)):
                ca += distances[i]/dmax * (time_together[i]/tsum)

            return 1-ca
        else:
            return 0.0



    def compute(self) -> pd.DataFrame:
        """
        Computes pairwise time-based Ca for all user permutations.

        Args:
            distance (int): Distance threshold in meters.

        Returns:
            pd.DataFrame: Result dataframe with
                columns [A, B, MaxTime, TimeTogether, Ca]
        """
        users = self._get_users()
        pairs = list(permutations(users, 2))
        results_frame = pd.DataFrame(
            columns=["A", "B", "Ca"]
        )
        for users in pairs:
            ca = self._calc_meetings_values(
                users[0], users[1]
            )
            results_frame = pd.concat(
                [
                    results_frame,
                    pd.DataFrame(
                        [[users[0], users[1], ca]],
                        columns=["A", "B", "Ca"],
                    ),
                ],
                ignore_index=True,
            )
        results_frame.sort_values('Ca')
        return results_frame

    def compute2(self) -> pd.DataFrame:
        """
        Computes pairwise time-based Ca for all user permutations using parallel processing.

        Returns:
            pd.DataFrame: DataFrame with columns [A, B, Ca]
        """
        users = self._get_users()
        pairs = list(combinations(users, 2))
        data = self.data.copy(deep=True)

        # Zbuduj argumenty jako krotki (data, pair)
        args_list = [(data, pair) for pair in pairs]

        results = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(meeting_worker, args) for args in args_list]
            for future in as_completed(futures):
                results.append(future.result())

        results_frame = pd.DataFrame(results, columns=["A", "B", "Ca"])
        results_frame = results_frame.sort_values("Ca", ascending=False).reset_index(drop=True)
        return results_frame

def meeting_worker(args):
    data, pair = args
    a, b = pair
    try:
        ca = calc_meeting_wrapper(data, a, b)
        return (a, b, ca)
    except Exception as e:
        return (a, b, 0)  # Możesz dać NaN


def calc_meeting_wrapper(data, main_agent, second_agent):
    m_agent_data = data[data["user_id"] == main_agent].sort_values("datetime").copy()
    m_agent_data["start"] = m_agent_data["datetime"]
    m_agent_data["end"] = m_agent_data["datetime"].shift(-1)

    s_agent_data = data[data["user_id"] == second_agent].sort_values("datetime").copy()
    s_agent_data["start"] = s_agent_data["datetime"]
    s_agent_data["end"] = s_agent_data["datetime"].shift(-1)

    # Usuń zerowe przedziały
    m_agent_data = m_agent_data[m_agent_data["start"] < m_agent_data["end"]]
    s_agent_data = s_agent_data[s_agent_data["start"] < s_agent_data["end"]]

    distances = []
    time_together = []

    for _, row in m_agent_data.iterrows():
        df_time_sort = s_agent_data[
            (s_agent_data["start"] < row["end"])
            & (s_agent_data["end"] > row["start"])
        ]

        for _, row_sort in df_time_sort.iterrows():
            distance = geodesic(
                (row_sort["lat"], row_sort["lon"]),
                (row["lat"], row["lon"])
            ).meters

            common_start = max(row["start"], row_sort["start"])
            common_end = min(row["end"], row_sort["end"])
            period = (common_end - common_start).total_seconds()

            if period > 0:
                distances.append(distance)
                time_together.append(period)

    ca = 0.0
    if distances and time_together:
        dmax = max(distances)
        tsum = np.sum(time_together)
        for i in range(len(time_together)):
            ca += distances[i]/dmax * (time_together[i]/tsum)
        return 1 - ca
    else:
        return 0.0