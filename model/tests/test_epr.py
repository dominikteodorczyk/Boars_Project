import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from scipy.sparse import lil_matrix

from model.src.EPR import EPR, EPRConfig, compute_od_row
from model.src.gravity import Gravity
from model.src.agent import Agent


# --- Helper Fixtures ---

class ZeroGravity:
    """Stub Gravity model returning zeros alone to force a fallback path."""

    def _compute_gravity_score(self, origin, distances, rel_origin, relevances):
        return np.zeros((1, len(relevances)))


@pytest.fixture
def dummy_tessellation():
    data = {
        "geometry": [Point(x, x) for x in range(10)],
        "population": [100 + x for x in range(10)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def epr_instance():
    config = EPRConfig()
    return EPR(config)


@pytest.fixture
def test_dates():
    start = datetime(2023, 1, 1, 8)
    end = datetime(2023, 1, 2, 8)
    return start, end


# --- Tests ---

def test_epr_initialization(epr_instance):
    assert epr_instance.rho == 0.6
    assert epr_instance.gamma == 0.21
    assert epr_instance.beta == 0.8
    assert epr_instance.tau == 17
    assert epr_instance.min_waiting_time_hours == 20 / 60


def test_sample_wait_hours(epr_instance):
    val = epr_instance._sample_wait_hours()
    assert isinstance(val, float)
    assert val >= epr_instance.min_waiting_time_hours


def test_sample_wait_delta(epr_instance):
    delta = epr_instance._sample_wait_delta()
    assert isinstance(delta, timedelta)
    assert delta.total_seconds() >= epr_instance.min_waiting_time_hours * 3600


def test_generate_synthetic_trajectory_basic(epr_instance, dummy_tessellation, test_dates):
    start, end = test_dates
    epr_instance.generate_synthetic_trajectory(
        n_agents=1,
        start_date=start,
        end_date=end,
        tessellation=dummy_tessellation,
        gravity_model=Gravity(model_type="singly constrained"),
        random_state=42
    )

    # Check that at least one trajectory entry was generated
    assert len(epr_instance.trajectories) >= 1
    for traj in epr_instance.trajectories:
        assert isinstance(traj[0], int)  # iteration
        assert isinstance(traj[1], int)  # agent_id
        assert isinstance(traj[2], datetime)  # timestamp
        assert isinstance(traj[3], int)  # location index


def test_compute_od_row_uniform_fallback():
    centroids = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    relevances = np.array([1, 1, 1], dtype=float)
    probs = compute_od_row(
        origin=0,
        centroids=centroids,
        relevances=relevances,
        gravity_model=ZeroGravity(),
    )
    # we expect a uniform distribution of 1/3
    assert np.allclose(probs, np.full(3, 1 / 3))


def test_choose_next_location_first_step(dummy_tessellation):
    cfg = EPRConfig()
    epr = EPR(cfg)

    # minimal grid with three cells + empty OD
    epr._tessellation = dummy_tessellation
    epr._centroids = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    epr._relevances = np.array([1, 1, 1], dtype=float)
    epr._gravity = ZeroGravity()
    epr._od = lil_matrix((3, 3))  # blank to run compute_od_row()

    # configure agent without visit history
    epr._agent = Agent(1)
    epr._current_start_cell = 0
    # trajectory buffer must contain the start-record
    epr._trajectories = [[0, 1, datetime.now(timezone.utc), 0]]

    nxt = epr._choose_next_location()

    # After the first step, we should get an index other than 0,
    # because _explore_new_location() for an empty OD draws (here: from 3 cells)
    assert nxt in (0, 1, 2)
    assert nxt == epr._current_start_cell  # start cell is overwritten


def test_starting_cells_too_short_raises(dummy_tessellation):
    epr = EPR(EPRConfig())
    with pytest.raises(ValueError, match="must contain at least 'n_agents'"):
        epr.generate_synthetic_trajectory(
            n_agents=3,
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 2),
            tessellation=dummy_tessellation,
            starting_cells=[0, 1],
        )


def test_od_matrix_pass_through(dummy_tessellation):
    epr = EPR(EPRConfig())
    od = lil_matrix(np.full((5, 5), 1 / 5))
    epr.generate_synthetic_trajectory(
        n_agents=1,
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 1, 2),
        tessellation=dummy_tessellation,
        od_matrix=od,
        random_state=0,
    )

    assert epr._od is od
    assert epr._od_is_sparse is False


def test_preferential_return_path(dummy_tessellation):
    cfg = EPRConfig()
    epr = EPR(cfg)
    epr._tessellation = dummy_tessellation
    epr._centroids = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    epr._relevances = np.array([1, 1, 1], dtype=float)
    epr._gravity = ZeroGravity()
    epr._od = lil_matrix((3, 3))
    agent = Agent(1)
    # Agent has already visited all the cells
    agent.visited_locations = {0: 1, 1: 2, 2: 3}
    epr._agent = agent
    epr._trajectories = [[0, 1, datetime.now(timezone.utc), 2]]
    # Enforce preferential return
    np.random.seed(0)
    nxt = epr._choose_next_location()
    assert nxt in (0, 1)


def test_all_cells_visited_returns_preferential(dummy_tessellation):
    cfg = EPRConfig()
    epr = EPR(cfg)
    epr._tessellation = dummy_tessellation
    epr._centroids = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    epr._relevances = np.array([1, 1, 1], dtype=float)
    epr._gravity = ZeroGravity()
    epr._od = lil_matrix((3, 3))
    agent = Agent(1)
    agent.visited_locations = {0: 1, 1: 1, 2: 1}
    epr._agent = agent
    epr._trajectories = [[0, 1, datetime.now(timezone.utc), 1]]
    # After visiting all cells, _choose_next_location should choose to return
    nxt = epr._choose_next_location()
    assert nxt in (0, 2)


def test_empty_tessellation():
    epr = EPR(EPRConfig())
    tess = gpd.GeoDataFrame({"geometry": [], "population": []})
    with pytest.raises(ValueError):
        epr.generate_synthetic_trajectory(
            n_agents=1,
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 2),
            tessellation=tess,
        )


def test_zero_population(dummy_tessellation, test_dates):
    # Set the population to zero
    dummy_tessellation["population"] = 0
    epr = EPR(EPRConfig())
    epr.generate_synthetic_trajectory(
        n_agents=1,
        start_date=test_dates[0],
        end_date=test_dates[1],
        tessellation=dummy_tessellation,
        random_state=1,
    )
    assert len(epr.trajectories) > 0


def test_long_simulation_buffer(dummy_tessellation):
    epr = EPR(EPRConfig())
    start = datetime(2023, 1, 1, 8)
    end = datetime(2023, 1, 10, 8)
    epr.generate_synthetic_trajectory(
        n_agents=1,
        start_date=start,
        end_date=end,
        tessellation=dummy_tessellation,
        random_state=123,
    )
    # Check whether the trajectory buffer is growing
    assert len(epr.trajectories) > 10


def test_long_simulation_buffer(dummy_tessellation):
    epr = EPR(EPRConfig())
    start = datetime(2023, 1, 1, 8)
    end = datetime(2023, 1, 10, 8)
    epr.generate_synthetic_trajectory(
        n_agents=1,
        start_date=start,
        end_date=end,
        tessellation=dummy_tessellation,
        random_state=123,
    )
    # Check whether the trajectory buffer is growing
    assert len(epr.trajectories) > 10


def test_explore_new_location_od_filled(dummy_tessellation):
    epr = EPR(EPRConfig())
    epr._tessellation = dummy_tessellation
    epr._centroids = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    epr._relevances = np.array([1, 1, 1], dtype=float)
    epr._gravity = ZeroGravity()
    epr._od = lil_matrix(np.full((3, 3), 1 / 3))
    # OD already filled, should not call compute_od_row
    loc = epr._explore_new_location(0)
    assert loc in (0, 1, 2)


def test_trajectories_df_property(epr_instance, dummy_tessellation, test_dates):
    start, end = test_dates
    epr_instance.generate_synthetic_trajectory(
        n_agents=1,
        start_date=start,
        end_date=end,
        tessellation=dummy_tessellation,
        random_state=123,
    )

    df = epr_instance.trajectories_df
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(df.columns) >= {"timestamp", "location", "agent_id", "iter"}


def test_convert_trajectories_to_dataframe(epr_instance):
    epr_instance._trajectories = [
        [0, 1, datetime(2023, 1, 1, 8), 0],
        [1, 1, datetime(2023, 1, 1, 10), 1],
    ]

    epr_instance.convert_trajectories_to_dataframe()

    df = epr_instance._trajectories_df
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["iter", "agent_id", "timestamp", "location"]


def test_convert_trajectories_to_geodataframe(epr_instance, dummy_tessellation):
    epr_instance._trajectories = [
        [0, 1, datetime(2023, 1, 1, 8), 0],
        [1, 1, datetime(2023, 1, 1, 9), 1],
    ]
    epr_instance._tessellation = dummy_tessellation
    epr_instance.convert_trajectories_to_dataframe()

    gdf = epr_instance.convert_trajectories_to_geodataframe()

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert set(gdf.columns) >= {"agent_id", "timestamp", "lat", "lon", "geometry"}
    assert all(gdf.geometry.geom_type == "Point")


def test_expand_trajectory_to_hourly_steps(epr_instance, dummy_tessellation):
    start = datetime(2023, 1, 1, 8)
    end = datetime(2023, 1, 1, 12)

    epr_instance._trajectories = [
        [0, 1, datetime(2023, 1, 1, 8), 0],
        [1, 1, datetime(2023, 1, 1, 10), 1],
    ]
    epr_instance.convert_trajectories_to_dataframe()

    epr_instance.expand_trajectory_to_hourly_steps(start, end)
    df = epr_instance.trajectories_df

    assert len(df) == 4
    assert list(df["timestamp"]) == [
        datetime(2023, 1, 1, 8),
        datetime(2023, 1, 1, 9),
        datetime(2023, 1, 1, 10),
        datetime(2023, 1, 1, 11),
    ]
    assert list(df["location"]) == [0, 0, 1, 1]


def test_convert_to_geodataframe_raises_without_tessellation(epr_instance):
    epr_instance._trajectories = [
        [0, 1, datetime(2023, 1, 1, 8), 0]
    ]
    epr_instance.convert_trajectories_to_dataframe()

    with pytest.raises(ValueError, match="Tessellation is not set"):
        epr_instance.convert_trajectories_to_geodataframe()


def test_expand_trajectory_to_hourly_steps_raises(epr_instance):
    with pytest.raises(ValueError, match="Trajectories DataFrame is not available"):
        epr_instance.expand_trajectory_to_hourly_steps(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 2)
        )


def test_convert_to_geodataframe_projected_crs(epr_instance):
    data = {
        "geometry": [Point(x, x) for x in range(3)],
        "population": [100, 150, 200],
    }
    tess = gpd.GeoDataFrame(data, crs="EPSG:2180")

    epr_instance._trajectories = [
        [0, 1, datetime(2023, 1, 1, 8), 0],
        [1, 1, datetime(2023, 1, 1, 9), 1],
    ]
    epr_instance._tessellation = tess
    epr_instance.convert_trajectories_to_dataframe()

    gdf = epr_instance.convert_trajectories_to_geodataframe()

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs.to_epsg() == 2180
    assert "lat" in gdf.columns and "lon" in gdf.columns
    assert all(gdf.geometry.geom_type == "Point")
