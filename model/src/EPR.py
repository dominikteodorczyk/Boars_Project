from datetime import datetime, timedelta

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import powerlaw
from tqdm import tqdm
from geopandas import GeoDataFrame
from scipy.sparse import lil_matrix
from pydantic import BaseModel, Field

from model.src.agent import Agent
from model.src.gravity import Gravity
from model.src.markov_chain import MarkovChain
from model.utils.utils import euclidean_distance, get_geom_centroid


def compute_od_row(origin: int, centroids: np.ndarray, relevances: np.ndarray,
                   gravity_model: Gravity) -> np.ndarray:
    origin_ll = centroids[origin]
    distances = np.array([euclidean_distance(origin_ll, dest_ll) for dest_ll in centroids])

    # Gravity scores for this origin to all destinations
    scores = gravity_model._compute_gravity_score(
        origin, distances, relevances[origin: origin + 1], relevances
    )[0]

    if scores.sum() == 0:
        # Fallback to a uniform distribution instead of zeros only.
        scores = np.ones_like(scores)

    return scores / scores.sum()


class EPRConfig(BaseModel):
    name: str = Field(default="EPR Model", description="Name of the EPR model")
    rho: float = Field(default=0.6, ge=0.0, le=1.0, description="Base exploration probability")
    gamma: float = Field(default=0.21, ge=0.0, description="Exploration decay rate")
    beta: float = Field(default=0.8, ge=0.0, description="Waiting‑time exponent")
    tau: int = Field(default=17, ge=1, description="Waiting‑time truncation (h)")
    min_waiting_time_minutes: int = Field(default=20, ge=0, description="Minimum waiting time in minutes")
    tessellation_attractiveness_column: str = Field(
        default="population",
        description="Column name in tessellation for spatial relevance (e.g., population, points_count)"
    )


class EPR:
    """Exploration & Preferential Return (EPR) mobility model.

    Implements the mechanism described in *Song et al., 2010* with an optional
    on‑demand gravity‑based OD matrix.
    """

    def __init__(self, config: EPRConfig) -> None:
        self.name = config.name

        # Hyper-parameters
        self._rho: float = config.rho
        self._gamma: float = config.gamma
        self._beta: float = config.beta
        self._tau: int = config.tau
        self._min_waiting_time_hours: float = config.min_waiting_time_minutes / 60.0

        # Spatial context
        self._tessellation: gpd.GeoDataFrame | None = None
        self._centroids: np.ndarray | None = None
        self._relevances: np.ndarray | None = None
        self._tessellation_attractiveness_column = config.tessellation_attractiveness_column

        # OD matrix (lazy)
        self._od: lil_matrix | None = None
        self._od_is_sparse: bool = True

        # Agent state (updated per‑agent iteration)
        self._current_start_cell: int | None = None
        self._agent: Agent | None = None
        self._gravity: Gravity | None = None

        # Output trajectory buffer: [iter, agent_id, timestamp, location]
        self._trajectories: list[list] = []
        self._trajectories_df: pd.DataFrame | None = None

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def tau(self) -> int:
        return self._tau

    @property
    def min_waiting_time_hours(self) -> float:
        return self._min_waiting_time_hours

    @property
    def trajectories(self) -> list[list]:
        return self._trajectories

    @property
    def trajectories_df(self) -> pd.DataFrame | None:
        return self._trajectories_df

    def convert_trajectories_to_dataframe(self) -> None:
        """Convert the trajectory list to a pandas DataFrame."""
        df = pd.DataFrame(
            self._trajectories,
            columns=["iter", "agent_id", "timestamp", "location"]
        )
        self._trajectories_df = df

    def convert_trajectories_to_geodataframe(self) -> gpd.GeoDataFrame:
        if self._tessellation is None:
            raise ValueError("Tessellation is not set. Cannot convert trajectories to GeoDataFrame.")
        if self._tessellation.crs.is_geographic:
            projected_tessellation = self._tessellation.to_crs(epsg=3857)
        else:
            projected_tessellation = self._tessellation
        centroids = np.array(projected_tessellation.geometry.centroid)
        self._trajectories_df["geometry"] = centroids[self._trajectories_df["location"].values]
        gdf = gpd.GeoDataFrame(self._trajectories_df, geometry="geometry", crs=self._tessellation.crs)
        gdf["lat"] = gdf.geometry.y
        gdf["lon"] = gdf.geometry.x

        return gdf[["agent_id", "timestamp", "lat", "lon", "geometry"]]

    def expand_trajectory_to_hourly_steps(self, start_date: datetime, end_date: datetime):
        """Expand the trajectory to hourly steps between start_date and end_date."""
        if self._trajectories_df is None:
            raise ValueError("Trajectories DataFrame is not available. Cannot expand to hourly steps.")
        df = self._trajectories_df.sort_values(["agent_id", "timestamp"])

        expanded = []

        for agent_id, group in df.groupby("agent_id"):
            group = group.set_index("timestamp")[["location"]]

            resampled = group.resample("1h").ffill()

            all_hours = pd.date_range(start=start_date, end=end_date, freq="1h", inclusive='left')
            resampled = resampled.reindex(all_hours, method="ffill")
            resampled = resampled.reset_index().rename(columns={"index": "timestamp"})

            resampled["agent_id"] = agent_id
            resampled["iter"] = range(len(resampled))
            expanded.append(resampled)

        self._trajectories_df = (
            pd.concat(expanded, ignore_index=True)
            if expanded else
            pd.DataFrame(columns=["timestamp", "location", "agent_id", "iter"])
        )

    def _sample_wait_hours(self) -> float:
        """Draw a single waiting time (in hours) from a truncated power law."""
        dist = powerlaw.Truncated_Power_Law(
            xmin=self._min_waiting_time_hours, parameters=[1.0 + self._beta, 1.0 / self._tau]
        )
        return float(dist.generate_random()[0])

    def _sample_wait_delta(self) -> timedelta:
        """Return waiting time as ``datetime.timedelta``."""
        return timedelta(hours=self._sample_wait_hours())

    def _pick_preferential_location(self, current_loc: int) -> int:
        """Select a *visited* cell according to visitation frequencies."""
        visited = self._agent.visited_locations
        locs, counts = zip(*((loc, cnt) for loc, cnt in visited.items() if loc != current_loc))
        probs = np.asarray(counts, dtype=float)
        probs /= probs.sum()
        return int(np.random.choice(locs, p=probs))

    def _explore_new_location(self, current_loc: int) -> int:
        """Pick a *new* location using (cached) OD probabilities."""
        # Retrieve or lazily compute probability vector for this origin.
        od_row = self._od.getrowview(current_loc)
        if od_row.nnz == 0:
            probs = compute_od_row(current_loc, self._centroids, self._relevances,
                                   self._gravity)  # type: ignore[arg-type]
            self._od[current_loc, :] = probs
        else:
            probs = od_row.toarray().ravel()

        destinations = np.arange(len(self._centroids))
        return int(np.random.choice(destinations, p=probs))

    def _choose_next_location(self) -> int:
        """Decide whether to *explore* or *return*, then pick the destination."""
        num_visited = len(self._agent.visited_locations)
        # First step: explore from the starting cell
        if num_visited == 0:
            self._current_start_cell = self._explore_new_location(self._current_start_cell)
            return self._current_start_cell

        *_prev, current_loc = self._trajectories[-1]

        # Exploration probability decays with #visited locations (Song et al.)
        p_new = np.random.random()
        threshold = self._rho * math.pow(num_visited, -self._gamma)
        explore_flag = (p_new <= threshold and num_visited != self._od.shape[0]) or num_visited == 1

        if explore_flag:
            return self._explore_new_location(current_loc)
        return self._pick_preferential_location(current_loc)

    def generate_synthetic_trajectory(self, n_agents: int, start_date: datetime, end_date: datetime,
                                      tessellation: gpd.GeoDataFrame, gravity_model: Gravity | None = None,
                                      starting_cells: list[int] | None = None, od_matrix: lil_matrix | None = None,
                                      random_state: int | None = None) -> GeoDataFrame:
        """Simulate trajectories for *n_agents* between *start_date* and *end_date*."""
        if starting_cells is not None and len(starting_cells) < n_agents:
            raise ValueError("'starting_cells' must contain at least 'n_agents' elements.")

        if random_state is not None:
            np.random.seed(random_state)

        # Spatial context
        self._tessellation = tessellation
        n_cells = len(self._tessellation)

        self._relevances = self._tessellation[self._tessellation_attractiveness_column].fillna(0).values
        self._centroids = self._tessellation.geometry.apply(get_geom_centroid, args=[True]).values

        # Gravity model (lazily created)
        self._gravity = gravity_model or Gravity(model_type="singly constrained")

        # OD matrix initialisation
        if od_matrix is None:
            self._od = lil_matrix((n_cells, n_cells))
            self._od_is_sparse = True
        else:
            self._od = od_matrix
            self._od_is_sparse = False

        # Simulate each agent separately
        with tqdm(total=n_agents, desc="Trajectory generation") as progress:
            for agent_id in range(1, n_agents + 1):
                self._current_start_cell = (
                    starting_cells.pop() if starting_cells is not None else np.random.randint(0, n_cells)
                )
                self._agent = Agent(agent_id)
                self._simulate_agent_trajectory(self._agent, start_date, end_date)
                progress.update(1)
        self.convert_trajectories_to_dataframe()
        self.expand_trajectory_to_hourly_steps(start_date, end_date)
        return self.convert_trajectories_to_geodataframe()

    def _simulate_agent_trajectory(self, agent: Agent, start_date: datetime, end_date: datetime) -> None:
        """Run the event‑based simulation loop for a single agent."""
        current_time = start_date
        iter_idx = 0

        # First record — agent starts at *start* timestamp
        self._trajectories.append([iter_idx, agent.id, current_time, self._current_start_cell])
        agent.visited_locations[self._current_start_cell] += 1

        # Iterate until *end_date*
        while True:
            current_time += self._sample_wait_delta()
            if current_time >= end_date:
                break

            iter_idx += 1
            next_loc = self._choose_next_location()
            agent.visited_locations[next_loc] += 1
            self._trajectories.append([iter_idx, agent.id, current_time, next_loc])


class Ditras(EPR):
    """Ditras model, inheriting from EPR."""

    def __init__(self, config: EPRConfig, diary_generator: MarkovChain) -> None:
        super().__init__(config)
        self._diary_generator = diary_generator

    def generate_synthetic_trajectory(self, n_agents: int, start_date: datetime, end_date: datetime,
                                      tessellation: gpd.GeoDataFrame, gravity_model: Gravity | None = None,
                                      starting_cells: list[int] | None = None, od_matrix: lil_matrix | None = None,
                                      random_state: int | None = None) -> GeoDataFrame:
        """Simulate trajectories for *n_agents* between *start_date* and *end_date*."""
        if starting_cells is not None and len(starting_cells) < n_agents:
            raise ValueError("'starting_cells' must contain at least 'n_agents' elements.")

        if random_state is not None:
            np.random.seed(random_state)

        # Spatial context
        self._tessellation = tessellation
        n_cells = len(self._tessellation)

        self._relevances = self._tessellation[self._tessellation_attractiveness_column].fillna(0).values
        self._centroids = self._tessellation.geometry.apply(get_geom_centroid, args=[True]).values

        # Gravity model (lazily created)
        self._gravity = gravity_model or Gravity(model_type="singly constrained")

        # OD matrix initialisation
        if od_matrix is None:
            self._od = lil_matrix((n_cells, n_cells))
            self._od_is_sparse = True
        else:
            self._od = od_matrix
            self._od_is_sparse = False

        # Simulate each agent separately
        with tqdm(total=n_agents, desc="Trajectory generation") as progress:
            for agent_id in range(1, n_agents + 1):
                self._current_start_cell = (
                    starting_cells.pop() if starting_cells is not None else np.random.randint(0, n_cells)
                )
                self._agent = Agent(agent_id)
                self._simulate_agent_trajectory(self._agent, start_date, end_date)
                progress.update(1)
        self.convert_trajectories_to_dataframe()
        self.expand_trajectory_to_hourly_steps(start_date, end_date)
        return self.convert_trajectories_to_geodataframe()

    def _simulate_agent_trajectory(self, agent: Agent, start_date: datetime, end_date: datetime) -> None:
        """Run the event‑based simulation loop for a single agent."""

        n_hours = int((end_date - start_date).total_seconds() // 3600)
        iter_idx = 0

        diary_df = self._diary_generator.generate(duration_hours=n_hours, start_date=start_date,
                                                  seed=np.random.randint(0, 10 ** 6))

        for _, row in diary_df.iterrows():
            if row.abstract_location == 0:
                next_loc = self._current_start_cell
            else:
                iter_idx += 1
                next_loc = self._choose_next_location()

            agent.visited_locations[next_loc] += 1
            self._trajectories.append([iter_idx, agent.id, row.datetime, next_loc])
