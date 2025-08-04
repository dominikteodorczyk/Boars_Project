from collections import defaultdict, Counter
from typing import Tuple, List, Optional
from humobi.structures.trajectory import TrajectoriesFrame
from datetime import timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from pydantic import BaseModel, Field, field_validator


class MarkovChainConfig(BaseModel):
    name: str = Field(default="Markov Chain", description="Name of the Markov chain model")
    chain_length: int = Field(..., description="Length of the Markov chain (24, 168, or 336)")
    time_slot: str = Field(default="1h", description="Time slot granularity, e.g. '1h'")
    label_column: str = Field(default="labels", description="Name of the column with location labels")

    @field_validator("chain_length")
    @classmethod
    def validate_chain_length(cls, value: int) -> int:
        if value not in [24, 168, 336]:
            raise ValueError("chain_length must be one of: 24, 168, or 336")
        return value


class MarkovChain:

    def __init__(self, config: MarkovChainConfig) -> None:

        self.name = config.name
        self.chain_length = config.chain_length
        self.time_slot = config.time_slot
        self.label_column = config.label_column
        self.chain: defaultdict[tuple[int, int], defaultdict[tuple[int, int], float]] = defaultdict(
            lambda: defaultdict(float))

    def _initialize_empty_chain(self) -> None:
        states = [(h, r) for h in range(self.chain_length) for r in [0, 1]]
        self.chain = defaultdict(lambda: defaultdict(float))
        for state1 in states:
            for state2 in states:
                self.chain[state1][state2] = 0.0

    def _group_by_time_slot(self, trajectory: TrajectoriesFrame) -> pd.Series:
        series = trajectory[self.label_column].astype(str)
        grouped = series.groupby(pd.Grouper(freq=self.time_slot)).apply(','.join).replace('', np.nan)
        return grouped

    @staticmethod
    def _compute_location_stats(grouped_series: pd.Series) -> tuple[dict, dict]:
        freq: Counter[str] = Counter()
        for entry in grouped_series.dropna():
            freq.update(location.strip() for location in entry.split(','))
        ranks = {loc: i + 1 for i, (loc, _) in enumerate(freq.most_common())}
        return dict(freq), ranks

    @staticmethod
    def _most_frequent_location(slot_string: str, freq_map: dict) -> str | float:
        if not isinstance(slot_string, str) or not slot_string.strip():
            return np.nan

        locations = slot_string.split(',')
        location_counts = Counter(locations)

        if len(locations) == 1:
            return locations[0]

        if len(set(location_counts.values())) == 1:
            return max(locations, key=lambda loc: freq_map.get(loc, 0))

        return location_counts.most_common(1)[0][0]

    def _get_time_shift(self, date: pd.Timestamp) -> int:
        date_midnight = date.replace(hour=0, minute=0, second=0, microsecond=0)

        def start_of_week(weeks_back=1):
            # Returns the start of the week shifted back by n_weeks weeks (default is 1 week)
            return date_midnight - timedelta(days=date_midnight.weekday() + 7 * (weeks_back - 1))

        if self.chain_length == 24:
            return date.hour
        elif self.chain_length == 168:
            return int((date - start_of_week()).total_seconds() // 3600)
        elif self.chain_length == 336:
            is_even_week = date_midnight.isocalendar().week % 2 == 0
            start = start_of_week() if is_even_week else start_of_week(2)
            return int((date - start).total_seconds() // 3600)
        else:
            raise ValueError(f"Unsupported chain_length: {self.chain_length}")

    def _process_individual(self, user_data: TrajectoriesFrame) -> pd.Series:
        grouped = self._group_by_time_slot(user_data)
        freq_map, rank_map = self._compute_location_stats(grouped)
        abstract_series = grouped.apply(lambda x: self._most_frequent_location(x, freq_map))
        abstract_series.ffill(inplace=True)
        abstract_series.bfill(inplace=True)
        abstract_series = abstract_series.apply(lambda x: rank_map[x])
        return abstract_series

    @staticmethod
    def _compute_tau(series: pd.Series, start_idx: int, target_loc: int) -> Tuple[int, int]:

        n = len(series)
        tau = 1  # start with 1 to count the current location
        j = start_idx
        while j < n and series.iloc[j] == target_loc:
            tau += 1
            j += 1
        return tau, j

    def _handle_home_transition(self, series: pd.Series, slot: int, h: int, next_h: int, next_loc: int, n: int) -> int:
        home, non_typical = 1, 0
        if next_loc == home:
            self.chain[(h, home)][(next_h, home)] += 1
        else:
            if slot + 2 < n:
                tau, end_idx = self._compute_tau(series, slot + 2, next_loc)
                h_tau = (h + tau) % self.chain_length
                self.chain[(h, home)][(h_tau, non_typical)] += 1
                slot = end_idx - 2
            else:
                slot = n

        return slot

    def _handle_non_home_transition(self, series: pd.Series, slot: int, h: int, next_h: int, next_loc: int,
                                    n: int) -> int:
        home, non_typical = 1, 0
        if next_loc == home:
            self.chain[(h, non_typical)][(next_h, home)] += 1
        else:
            if slot + 2 < n:
                tau, end_idx = self._compute_tau(series, slot + 2, next_loc)
                h_tau = (h + tau) % self.chain_length
                self.chain[(h, non_typical)][(h_tau, non_typical)] += 1
                slot = end_idx - 2
            else:
                slot = n
        return slot

    def _update_chain(self, series: pd.Series, shift: int = 0) -> None:
        home, non_typical = 1, 0
        n = len(series)
        slot = 0
        while slot < n - 1:
            h = (slot + shift) % self.chain_length
            next_h = (h + 1) % self.chain_length
            current, next_loc = series.iloc[slot], series.iloc[slot + 1]
            if current == home:
                slot = self._handle_home_transition(series, slot, h, next_h, next_loc, n)
            else:
                slot = self._handle_non_home_transition(series, slot, h, next_h, next_loc, n)
            slot += 1

    def _normalize_chain(self) -> None:
        for origin, transitions in self.chain.items():
            total = sum(transitions.values())
            if total > 0:
                for dest in transitions:
                    transitions[dest] /= total

    def fit(self, trajectory: TrajectoriesFrame) -> None:
        self._initialize_empty_chain()
        users = trajectory.get_users().tolist()

        with tqdm(total=len(users), desc="Fitting Markov Chain") as progress:
            for user in users:
                user_data = trajectory.uloc(user)
                series = self._process_individual(user_data)
                shift = self._get_time_shift(user_data.index.min())
                self._update_chain(series, shift=shift)
                progress.update(1)
        self._normalize_chain()

    @staticmethod
    def _choose_weighted(weights: list[float]) -> int:
        return int(np.searchsorted(np.cumsum(weights), random.random()))

    def generate(self, duration_hours: int, start_date: pd.Timestamp, seed: Optional[int] = None) -> pd.DataFrame:
        if seed is not None:
            random.seed(seed)

        states = []
        slot = self._get_time_shift(start_date)
        prev_state = (slot, 1)
        states.append(prev_state)

        step, hours_generated = slot, 0

        while hours_generated < duration_hours:
            h = step % self.chain_length
            transitions = self.chain[prev_state]
            probs = list(transitions.values())

            next_state = (
                ((h + 1) % self.chain_length, prev_state[1])
                if sum(probs) == 0 else
                list(transitions.keys())[self._choose_weighted(probs)]
            )

            states.append(next_state)
            delta = (next_state[0] - h) % self.chain_length
            step += delta
            hours_generated += delta
            prev_state = next_state

        diary = self._states_to_diary(states, duration_hours, start_date)
        return pd.DataFrame(diary, columns=['datetime', 'abstract_location'])

    def _states_to_diary(self, states: list[tuple], max_length: int, start_time: pd.Timestamp) -> List[Tuple[pd.Timestamp, int]]:
        current_time = start_time
        prev_state = states[0]
        location_counter = 1
        log: List[Tuple[pd.Timestamp, int]] = [(current_time, 0)]

        for state in states[1:]:
            h, loc = state
            h_prev, loc_prev = prev_state

            if loc == 1:
                current_time += timedelta(hours=1)
                log.append((current_time, 0))
                location_counter = 1
            else:
                hours = (h - h_prev) % self.chain_length
                for _ in range(hours):
                    current_time += timedelta(hours=1)
                    log.append((current_time, location_counter))
                location_counter += 1

            prev_state = state

        # Simplify log by merging consecutive entries with same location
        simplified: List[Tuple[pd.Timestamp, int]] = []
        last_location = -1
        for t, loc in log[:max_length]:
            if loc != last_location:
                simplified.append((t, loc))
                last_location = loc

        return simplified
