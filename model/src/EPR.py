from datetime import datetime, timedelta

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import powerlaw
import rasterio
from rasterstats import zonal_stats
from tqdm import tqdm
from geopandas import GeoDataFrame
from scipy.sparse import lil_matrix
from pydantic import BaseModel, Field, model_validator
from typing import Optional

from model.src.agent import Agent
from model.src.gravity import Gravity
from model.src.markov_chain import MarkovChain
from model.utils.utils import euclidean_distance, get_geom_centroid


def compute_od_row(origin: int, centroids: np.ndarray, relevances: np.ndarray,
                   gravity_model: Gravity) -> np.ndarray:
    """
    Compute a single row of the OD matrix for a given origin using the gravity model.
    The result is a probability distribution over all possible destinations.

    Args:
        origin (int): Index of the origin cell.
        centroids (np.ndarray): Array of shape (n_cells,) with list of (lat, lon) centroids.
        relevances (np.ndarray): Array of shape (n_cells,) with cell relevances (e.g., population).
        gravity_model (Gravity): An instance of the Gravity model to compute scores.
    Returns:
        np.ndarray: Probability distribution over destinations from the origin.
    """
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
    """
    Configuration for the Exploration & Preferential Return (EPR) mobility model.
    Validates parameters and provides default values.

    Attributes:
        name (str): Name of the EPR model.
        rho (float): Base exploration probability (0 <= rho <= 1).
        gamma (float): Exploration decay rate (gamma >= 0).
        beta (float): Waiting‑time exponent (beta >= 0).
        tau (int): Waiting‑time truncation in hours (tau >= 1).
        min_waiting_time_minutes (int): Minimum waiting time in minutes (min_waiting_time_minutes >= 0).
        tessellation_attractiveness_column (str): Column name in tessellation for spatial relevance.
        simulation_with_attractiveness_raster (bool): Whether to use an attractiveness raster for simulation.
        attractiveness_raster_path (Optional[str]): Path to the attractiveness raster file.
    """
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
    simulation_with_attractiveness_raster: bool = Field(
        default=False,
        description="Whether to use an attractiveness raster for simulation (if available)"
    )
    attractiveness_raster_path: Optional[str] = Field(
        default=None,
        description="Path to the attractiveness raster file (if simulation_with_attractiveness_raster is True)"
    )

    @model_validator(mode="after")
    def check_attractiveness_raster(self) -> "EPRConfig":
        """
        Validate that if `simulation_with_attractiveness_raster` is True,
        then `attractiveness_raster_path` must be provided and non-empty.

        Raises:
            ValueError: If the conditions are not met.
        Returns:
            EPRConfig: The validated configuration instance.
        """
        if self.simulation_with_attractiveness_raster:
            if not self.attractiveness_raster_path or self.attractiveness_raster_path.strip() == "":
                raise ValueError(
                    "`attractiveness_raster_path` must be provided and non-empty when "
                    "`simulation_with_attractiveness_raster` is True."
                )
        return self


class EPR:
    """Exploration & Preferential Return (EPR) mobility model.

    Implements the mechanism described in *Song et al., 2010* with an optional
    on‑demand gravity‑based OD matrix.
    """

    def __init__(self, config: EPRConfig) -> None:
        """
        Initialize the EPR model with the given configuration.

        Args:
            config (EPRConfig): Configuration parameters for the EPR model.
        """
        self.name = config.name

        # Hyper-parameters
        self._rho: float = config.rho
        self._gamma: float = config.gamma
        self._beta: float = config.beta
        self._tau: int = config.tau
        self._min_waiting_time_hours: float = config.min_waiting_time_minutes / 60.0

        # Spatial context
        self._simulation_with_attractiveness_raster: bool = config.simulation_with_attractiveness_raster
        self._attractiveness_raster_path: Optional[str] = config.attractiveness_raster_path
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
        """
        Base exploration probability.

        Getter:
            **Returns:**
                float: The base exploration probability (0 <= rho <= 1).
        """
        return self._rho

    @property
    def gamma(self) -> float:
        """
        Exploration decay rate.

        Getter:
            **Returns:**
                float: The exploration decay rate (gamma >= 0).
        """
        return self._gamma

    @property
    def beta(self) -> float:
        """
        Waiting‑time exponent.

        Getter:
            **Returns:**
                float: The waiting‑time exponent (beta >= 0).
        """
        return self._beta

    @property
    def tau(self) -> int:
        """
        Waiting‑time truncation in hours.

        Getter:
            **Returns:**
                int: The waiting‑time truncation in hours (tau >= 1).
        """
        return self._tau

    @property
    def min_waiting_time_hours(self) -> float:
        """
        Minimum waiting time in hours.

        Getter:
            **Returns:**
                float: The minimum waiting time in hours (min_waiting_time_hours >= 0).
        """
        return self._min_waiting_time_hours

    @property
    def trajectories(self) -> list[list]:
        """
        The raw trajectory list. Each entry is a list of: [iter, agent_id, timestamp, location].

        Getter:
            **Returns:**
                list[list]: The raw trajectory data.
        """
        return self._trajectories

    @property
    def trajectories_df(self) -> pd.DataFrame | None:
        """
        The trajectory data as a pandas DataFrame with columns: iter, agent_id, timestamp, location.

        Getter:
            **Returns:**
                pd.DataFrame | None: The trajectory DataFrame, or None if not yet converted.
        """
        return self._trajectories_df

    def convert_trajectories_to_dataframe(self) -> None:
        """
        Convert the trajectory list to a pandas DataFrame. Stores the result in `self._trajectories_df`.
        """
        df = pd.DataFrame(
            self._trajectories,
            columns=["iter", "agent_id", "timestamp", "location"]
        )
        self._trajectories_df = df

    def convert_trajectories_to_geodataframe(self) -> gpd.GeoDataFrame:
        """
        Convert the trajectory DataFrame to a GeoDataFrame with point geometries at cell centroids.
        Requires that the tessellation is set and that the trajectories DataFrame is available.

        Returns:
            gpd.GeoDataFrame: The trajectory data as a GeoDataFrame with columns: agent_id, timestamp, lat, lon, geometry.
        Raises:
            ValueError: If the tessellation is not set or if the trajectories DataFrame is not available.
        """
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

    def expand_trajectory_to_hourly_steps(self, start_date: datetime, end_date: datetime) -> None:
        """
        Expand the trajectory to hourly steps between start_date and end_date.
        Fills in missing hours by forward-filling the last known location.

        Args:
            start_date (datetime): The start date of the simulation period.
            end_date (datetime): The end date of the simulation period.
        Raises:
            ValueError: If the trajectories DataFrame is not available.
        Returns:
            None: The method updates `self._trajectories_df` in place.
        """
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
        """
        Draw a single waiting time (in hours) from a truncated power law.

        Returns:
            float: A sampled waiting time in hours.
        """
        dist = powerlaw.Truncated_Power_Law(
            xmin=self._min_waiting_time_hours, parameters=[1.0 + self._beta, 1.0 / self._tau]
        )
        return float(dist.generate_random()[0])

    def _sample_wait_delta(self) -> timedelta:
        """
        Return waiting time as ``datetime.timedelta``. Uses `_sample_wait_hours()` internally.

        Returns:
            timedelta: A sampled waiting time as a timedelta object.
        """
        return timedelta(hours=self._sample_wait_hours())

    def _pick_preferential_location(self, current_loc: int) -> int:
        """
        Select a *visited* cell according to visitation frequencies. Excludes the current location.

        Args:
            current_loc (int): The agent's current location index.
        Returns:
            int: The selected location index.
        """
        visited = self._agent.visited_locations
        locs, counts = zip(*((loc, cnt) for loc, cnt in visited.items() if loc != current_loc))
        probs = np.asarray(counts, dtype=float)
        probs /= probs.sum()
        return int(np.random.choice(locs, p=probs))

    def _explore_new_location(self, current_loc: int) -> int:
        """
        Pick a *new* location using (cached) OD probabilities. If the OD row for the current location is not yet
        computed, it will be calculated on‑demand. Excludes the current location from possible destinations. If no other
        locations are available (e.g., only one cell in tessellation), returns the current location.

        Args:
            current_loc (int): The agent's current location index.
        Returns:
            int: The selected new location index. If no new location is available, returns `current_loc`.
        """
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
        """
        Decide whether to *explore* or *return*, then pick the destination. The exploration probability decays with
        the number of unique visited locations. If exploring, a new location is chosen using the OD matrix;
        if returning, a previously visited location is selected based on visitation frequency. If no new locations
        are available, the agent will return to a visited location.

        Returns:
            int: The selected next location index.
        """
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
        """
        Simulate trajectories for *n_agents* between *start_date* and *end_date*. Each agent starts from a specified or
        random cell in the tessellation. The OD matrix is computed on‑demand using the provided or default gravity
        model. The resulting trajectories are returned as a GeoDataFrame with point geometries at cell centroids.

        Args:
            n_agents (int): Number of agents to simulate.
            start_date (datetime): Start date of the simulation period.
            end_date (datetime): End date of the simulation period.
            tessellation (gpd.GeoDataFrame): Spatial tessellation with a geometry column and an attractiveness column.
            gravity_model (Gravity | None): An optional Gravity model instance. If None, a default singly-constrained model is used.
            starting_cells (list[int] | None): Optional list of starting cell indices for agents. If provided, must have at least `n_agents` elements. If None, starting cells are chosen randomly.
            od_matrix (lil_matrix | None): Optional precomputed OD matrix. If None, it will be computed on‑demand.
            random_state (int | None): Optional random seed for reproducibility.
        Returns:
            gpd.GeoDataFrame: The simulated trajectories with columns: agent_id, timestamp, lat, lon, geometry.
        Raises:
            ValueError: If `starting_cells` is provided but has fewer than `n_agents` elements.
        """
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
        """
        Run the event‑based simulation loop for a single agent. The agent starts at the specified starting cell and
        iteratively decides to explore new locations or return to previously visited ones, based on the EPR model's
        probabilities. The agent waits at each location for a time drawn from a truncated power law distribution before
        moving again. The trajectory is recorded in `self._trajectories`.

        Args:
            agent (Agent): The agent to simulate.
            start_date (datetime): The start date of the simulation period.
            end_date (datetime): The end date of the simulation period.
        Returns:
            None: The method updates `self._trajectories` in place.
        """
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

    pick_preferential_location = _pick_preferential_location  # Expose for docs
    explore_new_location = _explore_new_location  # Expose for docs
    choose_next_location = _choose_next_location  # Expose for docs
    simulate_agent_trajectory = _simulate_agent_trajectory  # Expose for docs



class Ditras(EPR):
    """
    Ditras model, inheriting from EPR. Uses a Markov chain for diary generation. Overrides the trajectory generation method to
    incorporate diary-based location choices. Requires a MarkovChain instance for diary generation.
    """

    def __init__(self, config: EPRConfig, diary_generator: MarkovChain) -> None:
        """
        Initialize the Ditras model with the given configuration and diary generator. Inherits from EPR.

        Args:
            config (EPRConfig): Configuration parameters for the EPR model.
            diary_generator (MarkovChain): An instance of MarkovChain for generating activity diaries.
        """
        super().__init__(config)
        self._diary_generator = diary_generator

    def generate_synthetic_trajectory(self, n_agents: int, start_date: datetime, end_date: datetime,
                                      tessellation: gpd.GeoDataFrame, gravity_model: Gravity | None = None,
                                      starting_cells: list[int] | None = None, od_matrix: lil_matrix | None = None,
                                      random_state: int | None = None) -> GeoDataFrame:
        """
        Simulate trajectories for *n_agents* between *start_date* and *end_date*. Each agent starts from a specified or
        random cell in the tessellation. The OD matrix is computed on‑demand using the provided or default gravity
        model. The resulting trajectories are returned as a GeoDataFrame with point geometries at cell centroids.

        Args:
            n_agents (int): Number of agents to simulate.
            start_date (datetime): Start date of the simulation period.
            end_date (datetime): End date of the simulation period.
            tessellation (gpd.GeoDataFrame): Spatial tessellation with a geometry column and an attractiveness column.
            gravity_model (Gravity | None): An optional Gravity model instance. If None, a default singly-constrained model is used.
            starting_cells (list[int] | None): Optional list of starting cell indices for agents. If provided, must have at least `n_agents` elements. If None, starting cells are chosen randomly.
            od_matrix (lil_matrix | None): Optional precomputed OD matrix. If None, it will be computed on‑demand.
            random_state (int | None): Optional random seed for reproducibility.
        Returns:
            gpd.GeoDataFrame: The simulated trajectories with columns: agent_id, timestamp, lat, lon, geometry.
        Raises:
            ValueError: If `starting_cells` is provided but has fewer than `n_agents` elements.
        """
        if starting_cells is not None and len(starting_cells) < n_agents:
            raise ValueError("'starting_cells' must contain at least 'n_agents' elements.")

        if random_state is not None:
            np.random.seed(random_state)

        # Spatial context
        self._tessellation = tessellation
        n_cells = len(self._tessellation)

        if self._attractiveness_raster_path is not None and self._simulation_with_attractiveness_raster:
            raster = rasterio.open(self._attractiveness_raster_path)
            stats = zonal_stats(tessellation, self._attractiveness_raster_path, stats=['mean'], nodata=raster.nodata)
            mean_values = [s['mean'] if s['mean'] is not None else 0 for s in stats]
            tessellation[self._tessellation_attractiveness_column] *= mean_values

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
        """
        Run the event‑based simulation loop for a single agent.
        The agent starts at the specified starting cell and follows an activity diary generated by a Markov chain.
        The trajectory is recorded in `self._trajectories`.
        Args:
            agent (Agent): The agent to simulate.
            start_date (datetime): The start date of the simulation period.
            end_date (datetime): The end date of the simulation period.
        Returns:
            None: The method updates `self._trajectories` in place.
        """

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

    simulate_agent_trajectory = _simulate_agent_trajectory  # Expose for docs
