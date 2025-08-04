import pytest
from datetime import datetime, timedelta
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.sparse import lil_matrix
import pandas as pd

from model.src.EPR import EPRConfig, Ditras


# --- Helper Fixtures ---

class ZeroGravity:
    """Stub Gravity model returning zeros alone to force a fallback path."""

    def _compute_gravity_score(self, origin, distances, rel_origin, relevances):
        return np.zeros((1, len(relevances)))


class DummyDiary:
    """Minimal MarkovChain stub producing a deterministic diary suitable for tests.

    The generated diary alternates between `abstract_location == 0` ("home") and
    `abstract_location == 1` ("away"), one entry per hour.
    """

    def generate(self, duration_hours: int, start_date: datetime, seed: int):
        rng = np.random.default_rng(seed)
        datetimes = [start_date + timedelta(hours=i) for i in range(duration_hours)]
        # Alternate 0 / 1 so we both stay home and move
        abs_loc = [i % 2 for i in range(duration_hours)]
        diary_df = pd.DataFrame({
            "datetime": datetimes,
            "abstract_location": abs_loc,
        })
        return diary_df


@pytest.fixture
def dummy_diary():
    return DummyDiary()


@pytest.fixture
def dummy_tessellation():
    data = {
        "geometry": [Point(x, x) for x in range(10)],
        "population": [100 + x for x in range(10)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def ditras_instance(dummy_diary):
    cfg = EPRConfig()
    return Ditras(cfg, diary_generator=dummy_diary)


@pytest.fixture
def test_dates():
    start = datetime(2023, 1, 1, 8)
    end = datetime(2023, 1, 2, 8)
    return start, end


def test_ditras_initialization(ditras_instance):
    """The Ditras instance should inherit hyper‑parameters from EPRConfig."""
    assert ditras_instance.rho == 0.6
    assert ditras_instance.gamma == 0.21
    assert ditras_instance.beta == 0.8
    assert ditras_instance.tau == 17


def test_generate_synthetic_trajectory_basic(ditras_instance, dummy_tessellation, test_dates):
    """A very short simulation run should create at least one trajectory row."""
    start, end = test_dates
    ditras_instance.generate_synthetic_trajectory(
        n_agents=1,
        start_date=start,
        end_date=end,
        tessellation=dummy_tessellation,
        gravity_model=ZeroGravity(),
        random_state=123,
    )

    assert len(ditras_instance.trajectories) >= 1
    first_row = ditras_instance.trajectories[0]
    # Columns: iter_idx, agent_id, timestamp, location
    assert first_row[0] == 0  # first iteration index is 0
    assert isinstance(first_row[3], int)  # location index


def test_diary_home_location_logic(ditras_instance, dummy_tessellation, test_dates):
    """Rows with `abstract_location == 0` should keep the agent at the start cell."""
    start, end = test_dates
    # Force only two diary rows: first at home, second away
    ditras_instance._diary_generator = DummyDiary()
    ditras_instance.generate_synthetic_trajectory(
        n_agents=1,
        start_date=start,
        end_date=end,
        tessellation=dummy_tessellation,
        gravity_model=ZeroGravity(),
        random_state=1,
    )

    traj = ditras_instance.trajectories
    start_cell = traj[0][3]
    # Every even row in DummyDiary has abstract_location==0 → same cell
    assert traj[0][3] == start_cell  # first row (home)
    if len(traj) > 2:
        assert traj[2][3] == start_cell  # third row (home again)


def test_starting_cells_too_short_raises(ditras_instance, dummy_tessellation, test_dates):
    start, end = test_dates
    with pytest.raises(ValueError, match="must contain at least 'n_agents'"):
        ditras_instance.generate_synthetic_trajectory(
            n_agents=2,
            start_date=start,
            end_date=end,
            tessellation=dummy_tessellation,
            starting_cells=[0],
        )


def test_od_matrix_pass_through(ditras_instance, dummy_tessellation, test_dates):
    start, end = test_dates
    od = lil_matrix(np.full((len(dummy_tessellation), len(dummy_tessellation)), 1 / len(dummy_tessellation)))
    ditras_instance.generate_synthetic_trajectory(
        n_agents=1,
        start_date=start,
        end_date=end,
        tessellation=dummy_tessellation,
        od_matrix=od,
        random_state=0,
    )
    assert ditras_instance._od is od
    assert ditras_instance._od_is_sparse is False


def test_long_simulation_buffer(ditras_instance, dummy_tessellation):
    start = datetime(2025, 1, 1, 0)
    end = datetime(2025, 1, 3, 0)  # 48 h
    last_timestamp = end - timedelta(hours=1)  # last hour
    ditras_instance.generate_synthetic_trajectory(
        n_agents=1,
        start_date=start,
        end_date=end,
        tessellation=dummy_tessellation,
        random_state=42,
    )
    assert len(ditras_instance.trajectories) > 0
    assert ditras_instance.trajectories[0][2] == start  # first timestamp
    assert ditras_instance.trajectories[-1][2] == last_timestamp  # last timestamp
    assert len(ditras_instance.trajectories) == 48  # one row per hour
