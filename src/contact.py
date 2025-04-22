import pandas as pd
from geopy.distance import geodesic
from datetime import timedelta
from itertools import combinations, permutations


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
                    am_value += 1
                    # m_value += temp_dataframe.shape[0] - final_dataframe.shape[0]
            else:
                m_value += 1

        return am_value, m_value

    def _calc_meetings_values_vector(
        self, main_agent, secound_agent, td: int, dd: int
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
                        geodesic((s_row[3], s_row[2]), (m_row[3], m_row[2])).meters
                        <= dd
                    ):
                        am_value += 1
                        break

        m_value = m_agent_data.shape[0] - am_value
        return am_value, m_value

    def _compute_pair_ca(self, main_agent, secound_agent, td: int, dd: int):
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

    def compute(self, temporal: int = 3600, distance: int = 1000):
        users = self._get_users()
        pairs = list(combinations(users, 2))
        results_frame = pd.DataFrame(columns=["A", "B", "Ca", "AB1", "AB2", "A", "B"])
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

    def __init__(self) -> None:
        super().__init__()

    def _calc_meetings_values(self, main_agent, secound_agent, dd: int) -> tuple:

        m_agent_data = self.data[self.data["user_id"] == main_agent].sort_values(
            "datetime"
        )
        m_agent_data["start"] = m_agent_data["datetime"]
        m_agent_data["end"] = m_agent_data["datetime"].shift(-1)

        s_agent_data = self.data[self.data["user_id"] == secound_agent].sort_values(
            "datetime"
        )
        s_agent_data["start"] = s_agent_data["datetime"]
        s_agent_data["end"] = s_agent_data["datetime"].shift(-1)

        time_together = timedelta(0)
        for _, row in m_agent_data.iterrows():
            df_time_sort = s_agent_data[
                (s_agent_data["start"] < row["end"])
                & (s_agent_data["end"] > row["start"])
            ]

            if df_time_sort.shape[0] != 0:
                for _sort, row_sort in df_time_sort.iterrows():
                    distance = geodesic(
                        (row_sort["lat"], row_sort["lon"]), (row["lat"], row["lon"])
                    ).meters
                    if distance <= dd:

                        common_start = max(row["start"], row_sort["start"])
                        common_end = min(row["end"], row_sort["end"])
                        period = common_end - common_start
                        time_together += period

        max_time = m_agent_data["end"].max() - m_agent_data["start"].min()
        ca = int(time_together.total_seconds()) / int(max_time.total_seconds())

        return max_time, time_together, ca

    def compute(self, distance: int = 1000):
        users = self._get_users()
        pairs = list(permutations(users, 2))
        results_frame = pd.DataFrame(
            columns=["A", "B", "MaxTime", "TimeTogether", "Ca"]
        )
        for users in pairs:
            max_time, time_together, ca = self._calc_meetings_values(
                users[0], users[1], distance
            )
            results_frame = pd.concat(
                [
                    results_frame,
                    pd.DataFrame(
                        [[users[0], users[1], max_time, time_together, ca]],
                        columns=["A", "B", "MaxTime", "TimeTogether", "Ca"],
                    ),
                ],
                ignore_index=True,
            )
        return results_frame
