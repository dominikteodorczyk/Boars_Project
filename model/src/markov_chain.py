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
    """
    Configuration for the Markov Chain model. Includes parameters for chain length, time slot granularity, and label
    column name.

    Attributes:
        name (str): Name of the Markov chain model.
        chain_length (int): Length of the Markov chain (24, 168, or 336).
        time_slot (str): Time slot granularity, e.g. '1h'.
        label_column (str): Name of the column with location labels.
    """
    name: str = Field(default="Markov Chain", description="Name of the Markov chain model")
    chain_length: int = Field(..., description="Length of the Markov chain (24, 168, or 336)")
    time_slot: str = Field(default="1h", description="Time slot granularity, e.g. '1h'")
    label_column: str = Field(default="labels", description="Name of the column with location labels")

    @field_validator("chain_length")
    @classmethod
    def validate_chain_length(cls, value: int) -> int:
        """
        Validate that chain_length is one of the accepted values: 24, 168, or 336.

        Args:
            value (int): The chain length to validate.
        Returns:
            int: The validated chain length.
        Raises:
            ValueError: If chain_length is not one of the accepted values.
        """
        if value not in [24, 168, 336]:
            raise ValueError("chain_length must be one of: 24, 168, or 336")
        return value


class MarkovChain:
    """
    A Markov Chain model for simulating individual mobility patterns based on historical trajectory data.
    The model uses a state space defined by time slots and abstract location types (home and non-typical locations).
    """

    def __init__(self, config: MarkovChainConfig) -> None:
        """
        Initialize the Markov Chain model with the given configuration.

        Args:
            config (MarkovChainConfig): Configuration parameters for the Markov Chain model.
        """

        self.name = config.name
        self.chain_length = config.chain_length
        self.time_slot = config.time_slot
        self.label_column = config.label_column
        self.chain: defaultdict[tuple[int, int], defaultdict[tuple[int, int], float]] = defaultdict(
            lambda: defaultdict(float))

    def _initialize_empty_chain(self) -> None:
        """
        Initialize an empty Markov chain with all possible states and zero transition probabilities.
        The state space consists of tuples (h, r) where h is the hour in the chain (0 to chain_length-1) and r is the
        location type (0 for non-typical, 1 for home).
        The transition probabilities are stored in a nested defaultdict structure.
        """
        states = [(h, r) for h in range(self.chain_length) for r in [0, 1]]
        self.chain = defaultdict(lambda: defaultdict(float))
        for state1 in states:
            for state2 in states:
                self.chain[state1][state2] = 0.0

    def _group_by_time_slot(self, trajectory: TrajectoriesFrame) -> pd.Series:
        """
        Group trajectory data into specified time slots, concatenating location labels within each slot.
        This method resamples the trajectory data based on the configured time slot and aggregates location labels by
        joining them with commas. Empty slots are represented as NaN.

        Args:
            trajectory (TrajectoriesFrame): The trajectory data to be grouped.
        Returns:
            pd.Series: A series indexed by time slots with concatenated location labels.
        """
        series = trajectory[self.label_column].astype(str)
        grouped = series.groupby(pd.Grouper(freq=self.time_slot)).apply(','.join).replace('', np.nan)
        return grouped

    @staticmethod
    def _compute_location_stats(grouped_series: pd.Series) -> tuple[dict, dict]:
        """
        Compute frequency and rank of locations from the grouped series.
        This method analyzes the concatenated location labels in each time slot to determine how often each location
        appears and assigns ranks based on frequency.

        Args:
            grouped_series (pd.Series): A series with concatenated location labels per time slot.
        Returns:
            tuple[dict, dict]: A tuple containing two dictionaries: one for location frequencies and another for location ranks.
        """
        freq: Counter[str] = Counter()
        for entry in grouped_series.dropna():
            freq.update(location.strip() for location in entry.split(','))
        ranks = {loc: i + 1 for i, (loc, _) in enumerate(freq.most_common())}
        return dict(freq), ranks

    @staticmethod
    def _most_frequent_location(slot_string: str, freq_map: dict) -> str | float:
        """
        Determine the most frequent location from a comma-separated string of locations.
        If there's a tie in frequency, the location with the highest overall frequency in freq_map is chosen.

        Args:
            slot_string (str): A comma-separated string of location labels.
            freq_map (dict): A dictionary mapping locations to their overall frequencies.
        Returns:
            str | float: The most frequent location label, or NaN if input is invalid.
        """
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
        """
        Calculate the time shift based on the date and chain length. This determines the starting hour in the Markov
        chain corresponding to the given date.

        Args:
            date (pd.Timestamp): The date and time to calculate the shift for.
        Returns:
            int: The calculated time shift (hour) in the Markov chain.
        Raises:
            ValueError: If chain_length is not one of the supported values (24, 168, 336).
        """
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
        """
        Process an individual's trajectory data to create a series of abstract locations per time slot.
        This method groups the user's trajectory data by time slots, computes location statistics, and fills in
        missing values by forward and backward filling. The resulting series maps each time slot to an abstract
        location rank.

        Args:
            user_data (TrajectoriesFrame): The trajectory data for a single user.
        Returns:
            pd.Series: A series indexed by time slots with abstract location ranks.
        """
        grouped = self._group_by_time_slot(user_data)
        freq_map, rank_map = self._compute_location_stats(grouped)
        abstract_series = grouped.apply(lambda x: self._most_frequent_location(x, freq_map))
        abstract_series.ffill(inplace=True)
        abstract_series.bfill(inplace=True)
        abstract_series = abstract_series.apply(lambda x: rank_map[x])
        return abstract_series

    @staticmethod
    def _compute_tau(series: pd.Series, start_idx: int, target_loc: int) -> Tuple[int, int]:
        """
        Compute the duration (tau) of consecutive occurrences of target_loc in the series starting from start_idx.
        Returns the count of consecutive occurrences and the index where the sequence ends.

        Args:
            series (pd.Series): The series to analyze.
            start_idx (int): The starting index to check for consecutive occurrences.
            target_loc (int): The location label to count consecutively.
        Returns:
            Tuple[int, int]: A tuple containing the count of consecutive occurrences (tau) and the index where the sequence ends.
        """

        n = len(series)
        tau = 1  # start with 1 to count the current location
        j = start_idx
        while j < n and series.iloc[j] == target_loc:
            tau += 1
            j += 1
        return tau, j

    def _handle_home_transition(self, series: pd.Series, slot: int, h: int, next_h: int, next_loc: int, n: int) -> int:
        """
        Handle transitions when the current location is 'home' (1). Updates the Markov chain based on the next location.

        Args:
            series (pd.Series): The series of abstract locations.
            slot (int): The current index in the series.
            h (int): The current hour in the Markov chain.
            next_h (int): The next hour in the Markov chain.
            next_loc (int): The next location label.
            n (int): The total length of the series.
        Returns:
            int: The updated index in the series after handling the transition.
        """
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
        """
        Handle transitions when the current location is 'non-typical' (not home). Updates the Markov chain based on the
        next location.

        Args:
            series (pd.Series): The series of abstract locations.
            slot (int): The current index in the series.
            h (int): The current hour in the Markov chain.
            next_h (int): The next hour in the Markov chain.
            next_loc (int): The next location label.
            n (int): The total length of the series.
        Returns:
            int: The updated index in the series after handling the transition.
        """

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
        """
        Update the Markov chain with transitions from the given series of abstract locations, applying a time shift.
        The method iterates through the series, handling transitions based on whether the current location is 'home' or
        'non-typical', and updates the transition counts in the Markov chain accordingly.

        Args:
            series (pd.Series): The series of abstract locations.
            shift (int): The time shift to apply to the series.
        Returns:
            None
        """
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
        """
        Normalize the transition counts in the Markov chain to probabilities.
        This method iterates through each origin state in the Markov chain and normalizes the transition counts to
        probabilities by dividing each count by the total count of transitions from that state. If a state has no
        outgoing transitions, its probabilities remain zero.

        Returns:
            None
        """
        for origin, transitions in self.chain.items():
            total = sum(transitions.values())
            if total > 0:
                for dest in transitions:
                    transitions[dest] /= total

    def fit(self, trajectory: TrajectoriesFrame) -> None:
        """
        Fit the Markov chain model to the provided trajectory data.
        The method initializes an empty Markov chain, processes each user's trajectory data to extract abstract location
        series, applies the appropriate time shift, updates the Markov chain with transitions, and finally normalizes
        the transition counts to probabilities.

        Args:
            trajectory (TrajectoriesFrame): The trajectory data to fit the model to.
        Returns:
            None
        """
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
        """
        Choose an index based on the provided weights using a weighted random selection. The method computes
        the cumulative sum of the weights and selects an index where a random number falls within the cumulative
        distribution.

        Args:
            weights (list[float]): A list of weights for each index.
        Returns:
            int: The selected index based on the weights.
        """
        return int(np.searchsorted(np.cumsum(weights), random.random()))

    def generate(self, duration_hours: int, start_date: pd.Timestamp, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate a synthetic trajectory for a specified duration starting from a given date using the fitted
        Markov chain. The method simulates the trajectory by transitioning through states in the Markov chain based
        on the learned transition probabilities. The generated trajectory is returned as a DataFrame with timestamps
        and abstract location labels.

        Args:
            duration_hours (int): The total duration of the generated trajectory in hours.
            start_date (pd.Timestamp): The starting date and time for the generated trajectory.
            seed (Optional[int]): An optional random seed for reproducibility.
        Returns:
            pd.DataFrame: A DataFrame containing the generated trajectory with columns 'datetime' and 'abstract_location'.
        """
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
        """
        Convert a list of states into a diary format with timestamps and location labels.
        This method processes the list of states generated by the Markov chain to create a diary that records the
        timestamp and corresponding location label for each hour. Consecutive entries with the same location are
        merged to simplify the diary.

        Args:
            states (list[tuple]): A list of states represented as tuples (hour, location).
            max_length (int): The maximum length of the diary in hours.
            start_time (pd.Timestamp): The starting timestamp for the diary.
        Returns:
            List[Tuple[pd.Timestamp, int]]: A list of tuples containing timestamps and location labels.
        """
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

    initialize_empty_chain = _initialize_empty_chain  # Expose for docs
    group_by_time_slot = _group_by_time_slot  # Expose for docs
    compute_location_stats = _compute_location_stats  # Expose for docs
    most_frequent_location = _most_frequent_location  # Expose for docs
    get_time_shift = _get_time_shift  # Expose for docs
    process_individual = _process_individual  # Expose for docs
    compute_tau = _compute_tau  # Expose for docs
    handle_home_transition = _handle_home_transition  # Expose for docs
    handle_non_home_transition = _handle_non_home_transition  # Expose for docs
    update_chain = _update_chain  # Expose for docs
    normalize_chain = _normalize_chain  # Expose for docs
    choose_weighted = _choose_weighted  # Expose for docs
    states_to_diary = _states_to_diary  # Expose for docs
