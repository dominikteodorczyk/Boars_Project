import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from humobi.structures.trajectory import TrajectoriesFrame
from model.src.markov_chain import MarkovChain
from model.src.markov_chain import MarkovChainConfig
from pydantic import ValidationError

import warnings

config_24 = MarkovChainConfig(chain_length=24, time_slot="1h", label_column="labels")
config_168 = MarkovChainConfig(chain_length=168, time_slot="1h", label_column="labels")
config_336 = MarkovChainConfig(chain_length=336, time_slot="1h", label_column="labels")


class DummyTrajectoriesFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return DummyTrajectoriesFrame


@pytest.fixture
def sample_trajectory():
    return TrajectoriesFrame(pd.DataFrame({
        "labels": ["1", "2", "1", "3", "2"],
        "timestamp": pd.date_range("2023-01-01", periods=5, freq="1H")
    }).set_index("timestamp"))


def test_initializes_chain_with_correct_length():
    mc = MarkovChain(config_24)
    mc._initialize_empty_chain()
    assert len(mc.chain) == 48  # 24 hours * 2 states (0, 1)


def test_invalid_chain_length_raises():
    with pytest.raises(ValidationError) as exc_info:
        MarkovChainConfig(chain_length=10)  # 10 is an incorrect value

    assert "chain_length must be one of: 24, 168, or 336" in str(exc_info.value)


def test_group_by_time_slot():
    data = {
        "labels": ["1", "2", "1", "3", "2"],
    }
    index = pd.date_range("2023-01-01 00:00", periods=5, freq="30min")
    df = DummyTrajectoriesFrame(data, index=index)

    mc = MarkovChain(config_24)
    result = mc._group_by_time_slot(df)

    expected_index = pd.date_range("2023-01-01 00:00", periods=3, freq="1h")
    expected = pd.Series(["1,2", "1,3", "2"], index=expected_index, name="labels")
    expected = expected.replace("", np.nan)

    pd.testing.assert_series_equal(result, expected)


def test_compute_location_stats():
    grouped = pd.Series(["1,2", "1,3", "2"], index=pd.date_range("2023-01-01 00:00", periods=3, freq="1h"))

    freq, rank = MarkovChain._compute_location_stats(grouped)

    expected_freq = {"1": 2, "2": 2, "3": 1}
    expected_rank = {"1": 1, "2": 2, "3": 3}

    assert dict(freq) == expected_freq
    assert dict(rank) == expected_rank


def test_most_frequent_location():
    # Case  1: only one location
    assert MarkovChain._most_frequent_location("1", {"1": 5}) == "1"

    # Case 2: multiple locations with same freq, choose one with higher overall freq
    assert MarkovChain._most_frequent_location("1,2", {"1": 3, "2": 2}) == "1"
    assert MarkovChain._most_frequent_location("1,2", {"1": 1, "2": 4}) == "2"

    # Case 3: different frequencies in the slot
    assert MarkovChain._most_frequent_location("1,1,2", {"1": 2, "2": 5}) == "1"
    assert MarkovChain._most_frequent_location("2,2,1,1,1", {"1": 1, "2": 1}) == "1"

    # Case 4: no locations (None or empty string)
    assert MarkovChain._most_frequent_location(None, {"A": 1}) is np.nan
    assert MarkovChain._most_frequent_location("", {"A": 1}) is np.nan


def test_get_time_shift():
    date = pd.Timestamp("2023-04-05 15:30")  # Wednesday, 15:30

    mc_24 = MarkovChain(config_24)
    assert mc_24._get_time_shift(date) == 15

    mc_168 = MarkovChain(config_168)
    # Monday 00:00 start of week, diff = 2 days 15 hours = 63 hours
    assert mc_168._get_time_shift(date) == 63

    # Chain case 336 (biweekly hours)
    mc_336 = MarkovChain(config_336)
    assert mc_336._get_time_shift(date) == 63

    date = pd.Timestamp("2023-04-12 15:30")
    mc_336 = MarkovChain(config_336)
    # 9 days 15 hours diff = 231
    assert mc_336._get_time_shift(date) == 231

    mc_24_valid = MarkovChain(config_24)
    mc_24_valid.chain_length = 25
    with pytest.raises(ValueError):
        mc_24_valid._get_time_shift(date)


def test_fit():
    data = {
        "labels": ["", "1", "", "2", "", "1", "", "1", "2"]
    }
    # Replace empty strings with np.nan
    pd.Series(data["labels"]).replace("", np.nan, inplace=True)
    index = pd.date_range("2023-04-03 00:00", periods=9, freq="1h")
    user_data = DummyTrajectoriesFrame(data, index=index)

    mc = MarkovChain(config_24)
    result = mc._process_individual(user_data)

    expected = pd.Series([1, 1, 1, 2, 2, 1, 1, 1, 2], index=pd.date_range("2023-04-03 00:00", periods=9, freq="1h"),
                         name="labels")
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize("time_series,expected", [

    (
            pd.Series([1, 1, 1]),
            [((0, 1), (1, 1), 1.0), ((1, 1), (2, 1), 1.0)],
    ),

    (
            pd.Series([1, 0, 0, 0, 1]),
            [((0, 1), (3, 0), 1.0), ((3, 0), (4, 1), 1.0)],
    ),

    (
            pd.Series([0, 1]),
            [((0, 0), (1, 1), 1.0)],
    ),

    (
            pd.Series([0, 0, 0, 1]),
            [((0, 0), (2, 0), 1.0), ((2, 0), (3, 1), 1.0)],
    ),

    (
            pd.Series([1, 1, 0, 0, 1, 0, 0, 1]),
            [
                ((0, 1), (1, 1), 1.0),
                ((1, 1), (3, 0), 1.0),
                ((3, 0), (4, 1), 1.0),
                ((4, 1), (6, 0), 1.0),
                ((6, 0), (7, 1), 1.0),
            ],
    ),

    (
            pd.Series([1, 0]),
            [],
    ),

    (
            pd.Series([1, 1, 1, 1]),
            [((0, 1), (1, 1), 1.0), ((1, 1), (2, 1), 1.0), ((2, 1), (3, 1), 1.0)],
    ),

    (
            pd.Series([1, 0, 0]),
            [],
    ),

    (
            pd.Series([0, 0]),
            [],
    ),

])
def test_update_chain(time_series, expected):
    mc = MarkovChain(config_24)
    mc._initialize_empty_chain()
    mc._update_chain(time_series)

    for from_state, to_state, count in expected:
        assert mc.chain[from_state][to_state] == count


def test_markov_chain_wraps_after_full_cycle():
    mc = MarkovChain(config_168)
    mc._initialize_empty_chain()

    time_series = pd.Series([1] * 85 + [0] * 100)

    mc._update_chain(time_series)

    TYPICAL, NON_TYPICAL = 1, 0
    from_state = (85 % mc.chain_length - 1, TYPICAL)
    to_state = ((85 + 100 - 1) % mc.chain_length, NON_TYPICAL)

    assert mc.chain[from_state][to_state] > 0.0


def test_normalize_chain():
    mc = MarkovChain(config_24)
    mc._initialize_empty_chain()

    state_from = (0, 1)
    mc.chain[state_from][(1, 1)] = 2.0
    mc.chain[state_from][(2, 1)] = 3.0
    mc.chain[state_from][(3, 0)] = 5.0

    mc.chain[(1, 1)][(2, 1)] = 1.0

    mc._normalize_chain()

    row1_sum = sum(mc.chain[state_from].values())
    assert abs(row1_sum - 1.0) < 1e-8, f"Sum for {state_from} = {row1_sum}, expected 1.0"

    assert mc.chain[state_from][(1, 1)] == pytest.approx(0.2)
    assert mc.chain[state_from][(2, 1)] == pytest.approx(0.3)
    assert mc.chain[state_from][(3, 0)] == pytest.approx(0.5)

    assert mc.chain[(1, 1)][(2, 1)] == 1.0


def test_process_data_updates_chain_correctly():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    df = pd.DataFrame({
        "labels": ["1", "1", "2", "1", "2", "2", "1", "1"],
    }, index=pd.MultiIndex.from_arrays([
        pd.date_range("2023-04-03 00:00", periods=8, freq="30min"),
        ["user1"] * 4 + ["user2"] * 4
    ], names=["datetime", "user"]))
    traj = TrajectoriesFrame(df)

    traj.get_users = MagicMock(return_value=np.array(["user1", "user2"]))
    traj.uloc = MagicMock(side_effect=lambda user: traj.xs(user, level="user"))

    mc = MarkovChain(config_24)
    mc._initialize_empty_chain()

    mc.fit(traj)

    chain = mc.chain

    total_transitions = sum(
        value for from_state in chain for value in chain[from_state].values()
    )
    assert total_transitions > 0, "Markov chain should have some transitions after fit"

    assert any(
        value > 0
        for from_state in chain
        for to_state, value in chain[from_state].items()
    ), "Markov chain transitions should have positive counts"


def test_choose_weighted():
    weights = [0.1, 0.2, 0.7]
    counts = [0, 0, 0]
    n = 10000

    for _ in range(n):
        idx = MarkovChain._choose_weighted(weights)
        counts[idx] += 1

    probs = [count / n for count in counts]

    assert abs(probs[0] - 0.1) < 0.02
    assert abs(probs[1] - 0.2) < 0.02
    assert abs(probs[2] - 0.7) < 0.02


def test_generate():
    mc = MarkovChain(config_24)
    mc.chain = {
        (0, 1): {(1, 1): 1.0},
        (1, 1): {(2, 1): 1.0},
        (2, 1): {(3, 1): 1.0},
        (3, 1): {(4, 1): 1.0},
        (4, 1): {(5, 1): 1.0},
        (5, 1): {(6, 1): 1.0},
        (6, 1): {(7, 1): 1.0},
        (7, 1): {(8, 1): 1.0},
        (8, 1): {(9, 1): 1.0},
        (9, 1): {(10, 1): 1.0},
        (10, 1): {(11, 1): 1.0},
        (11, 1): {(12, 1): 1.0},
        (12, 1): {(13, 1): 1.0},
        (13, 1): {(14, 1): 1.0},
        (14, 1): {(15, 1): 1.0},
        (15, 1): {(16, 1): 1.0},
        (16, 1): {(17, 1): 1.0},
        (17, 1): {(18, 1): 1.0},
        (18, 1): {(19, 1): 1.0},
        (19, 1): {(20, 1): 1.0},
        (20, 1): {(21, 1): 1.0},
        (21, 1): {(22, 1): 1.0},
        (22, 1): {(23, 1): 1.0},
        (23, 1): {(0, 1): 1.0}
    }
    start_date = pd.Timestamp('2024-01-01 00:00:00')
    diary_df = mc.generate(duration_hours=5, start_date=start_date, seed=42)
    assert len(diary_df) > 0
    assert all(isinstance(row['datetime'], pd.Timestamp) for _, row in diary_df.iterrows())
    assert all(row['abstract_location'] == 0 for _, row in diary_df.iterrows())


def test_generate_handles_zero_probabilities_and_advances_time():
    from model.src.markov_chain import MarkovChain
    mc = MarkovChain(config_24)
    mc.chain = {
        (0, 1): {(1, 1): 0.0, (2, 1): 0.0},
        (1, 1): {(2, 1): 0.0, (3, 1): 0.0},
        (2, 1): {(3, 1): 0.0, (4, 1): 0.0}
    }
    start_date = pd.Timestamp('2024-01-01 00:00:00')
    diary_df = mc.generate(duration_hours=2, start_date=start_date)
    assert len(diary_df) > 0


def test_generate_wrap_around_hours():
    mc = MarkovChain(config_24)
    mc.chain = {
        (22, 1): {(23, 1): 1.0},
        (23, 1): {(0, 1): 1.0},
        (0, 1): {(1, 1): 1.0},
    }
    start_date = pd.Timestamp('2024-01-01 22:00:00')
    diary_df = mc.generate(duration_hours=3, start_date=start_date)

    assert all(isinstance(row['datetime'], pd.Timestamp) for _, row in diary_df.iterrows())

    assert len(diary_df) >= 1


def test_generate_wrap_around_hours_with_loop():
    mc = MarkovChain(config_24)
    mc.chain = {
        (22, 1): {(1, 0): 1.0},
        (1, 0): {(2, 0): 1.0},
        (2, 0): {(3, 1): 1.0},
        (3, 1): {(4, 0): 1.0},
        (4, 0): {(5, 1): 1.0},

    }
    start_date = pd.Timestamp('2024-01-01 22:00:00')
    diary_df = mc.generate(duration_hours=6, start_date=start_date)

    assert len(diary_df) > 1

    hours_in_diary = [dt.hour for dt in diary_df['datetime']]
    expected_hours = [22, 23, 2, 3]
    assert all(h in hours_in_diary for h in expected_hours)
