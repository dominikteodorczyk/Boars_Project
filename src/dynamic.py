import copy
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from datetime import timedelta
from tqdm import tqdm
from copy import copy
from itertools import combinations


class CoeffAssociation:

    def __init__(self) -> None:
        self.data = pd.DataFrame()
        pass

    def _get_users(self) -> list:
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

        self.data = data.rename(
            columns={id_col: "user_id", timestamp: "datetime", lat: "lat", lon: "lon"}
        )
        self.data["datetime"] = pd.to_datetime(self.data["datetime"])


class EventCa(CoeffAssociation):

    def __init__(self) -> None:
        super().__init__()

    def _calc_meetings_values(
        self, main_agent, secound_agent, td: timedelta, dd: int
    ) -> tuple:
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

            filtred_dataframe = temp_dataframe[temp_dataframe["time_diff"] <= td]
            if filtred_dataframe.shape[0] > 0:
                filtred_dataframe = filtred_dataframe.copy()
                filtred_dataframe.loc[:, "distance"] = filtred_dataframe.apply(
                    lambda x: geodesic(
                        (x["lat"], x["lon"]), (x["m_lat"], x["m_lon"])
                    ).meters,
                    axis=1,
                )
                final_dataframe = filtred_dataframe[filtred_dataframe["distance"] <= dd]
                if final_dataframe.shape[0] > 0:
                    am_value += final_dataframe.shape[0]
                    m_value += temp_dataframe.shape[0] - final_dataframe.shape[0]
            else:
                m_value += temp_dataframe.shape[0]

        return am_value, m_value

    def _compute_pair_ca(self, main_agent, secound_agent, td: int, dd: int):
        time_threshold = timedelta(seconds=td)

        ab1, a_only = self._calc_meetings_values(
            main_agent, secound_agent, time_threshold, dd
        )
        ab2, b_only = self._calc_meetings_values(
            secound_agent, main_agent, time_threshold, dd
        )

        denominator = a_only + b_only
        if denominator > 0:
            return (2 * ab2) / denominator
        elif denominator == 0:
            return 1.0

    def compute(self, temporal: int = 3600, distance: int = 1000):
        users = self._get_users()
        pairs = list(combinations(users, 2))
        results_frame = pd.DataFrame(columns=["A", "B", "Ca"])
        for users in pairs:
            ca = self._compute_pair_ca(users[0], users[1], temporal, distance)
            results_frame = pd.concat(
                [
                    results_frame,
                    pd.DataFrame([[users[0], users[1], ca]], columns=["A", "B", "Ca"]),
                ],
                ignore_index=True,
            )
        return results_frame


class TimeCa(CoeffAssociation):

    def __init__(self) -> None:
        super().__init__()

    def _calc_meetings_values(
        self, main_agent, secound_agent, td: timedelta, dd: int
    ) -> tuple:

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

        m_full_time = m_agent_data['datetime'].max() - m_agent_data['datetime'].min()


        temp_dataframe["m_datetime"] = m_row["datetime"]
        temp_dataframe["m_lat"] = m_row["lat"]
        temp_dataframe["m_lon"] = m_row["lon"]
        temp_dataframe["time_diff"] = abs(
            temp_dataframe["datetime"] - temp_dataframe["m_datetime"]
        )


        # return am_value, m_value

    def compute(self, temporal: int = 3600, distance: int = 1000):
        users = self._get_users()
        pairs = list(combinations(users, 2))
        for users in pairs:
            ca = self._calc_meetings_values(users[0], users[1], temporal, distance)
