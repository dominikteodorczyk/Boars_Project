import numpy as np
import geopandas as gpd
from ..utils import utils
from typing import Callable, Sequence
from tqdm import tqdm


def exponential_deterrence(distances: np.ndarray, rate: float) -> np.ndarray:
    """
    Exponential deterrence function: exp(-rate * d) where d is distance.
    Handles zero distances naturally since exp(0) = 1.

    Args:
        distances (np.ndarray): Array of distances.
        rate (float): Decay rate parameter.
    Returns:
        np.ndarray: Deterrence values.
    """
    return np.exp(-rate * distances)


def power_law_deterrence(distances: np.ndarray, exponent: float) -> np.ndarray:
    """
    Power-law deterrence function: d ** exponent where d is distance.
    Safely handles zero distances by replacing them with a small epsilon value.

    Args:
        distances (np.ndarray): Array of distances.
        exponent (float): Exponent parameter (typically negative).
    Returns:
        np.ndarray: Deterrence values.
    """
    if np.any(distances == 0):
        distances = np.where(distances == 0, np.finfo(float).eps, distances)
    return np.power(distances, exponent)


DETERRENCE_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "power_law": power_law_deterrence,
    "exponential": exponential_deterrence,
}


def compute_distance_matrix(spatial_tessellation: gpd.GeoDataFrame, origins: Sequence[int], *,
                            centroid_func: Callable = utils.get_geom_centroid) -> np.ndarray:
    """
    Compute a symmetric distance matrix between the centroids of the spatial tessellation. Only computes
    distances for the specified origin indices to optimize performance. Uses the provided centroid function.

    Args:
        spatial_tessellation (gpd.GeoDataFrame): GeoDataFrame with geometries of spatial cells.
        origins (Sequence[int]): Indices of origin cells to compute distances for.
        centroid_func (Callable): Function to compute centroids from geometries. Defaults to utils.get_geom_centroid.
    Returns:
        np.ndarray: Symmetric distance matrix of shape (n_cells, n_cells).
    """
    centroids = np.array(
        spatial_tessellation.geometry.apply(centroid_func, args=[True]).tolist(),
        dtype=float,
    )
    n_cells = len(spatial_tessellation)
    dist_matrix = np.zeros((n_cells, n_cells), dtype=float)

    for origin_idx in tqdm(origins, desc="Computing centroid distances"):
        lat_i, lon_i = centroids[origin_idx]

        for dest_idx in range(origin_idx + 1, n_cells):
            lat_j, lon_j = centroids[dest_idx]
            distance = np.sqrt((lat_i - lat_j) ** 2 + (lon_i - lon_j) ** 2)
            dist_matrix[origin_idx, dest_idx] = dist_matrix[dest_idx, origin_idx] = distance
    return dist_matrix


class Gravity:
    """
    Gravity model for spatial interaction based on distance deterrence and location relevances.
    Currently, only the core gravity score computation is implemented.
    """
    def __init__(self, deterrence_type: str = 'power_law', deterrence_params: list = [-2.0],
                 origin_exponent: float = 1.0, destination_exponent: float = 1.0,
                 model_type: str = 'singly constrained',
                 name: str = "Gravity Model") -> None:
        """
        Initialize the Gravity model with specified parameters.

        Args:
            deterrence_type (str): Type of deterrence function ('power_law' or 'exponential').
            deterrence_params (list): Parameters for the deterrence function.
            origin_exponent (float): Exponent for origin relevance.
            destination_exponent (float): Exponent for destination relevance.
            model_type (str): Type of gravity model, currently only 'singly constrained' is supported.
            name (str): Name of the model instance.
        Raises:
            ValueError: If an unsupported deterrence_type is provided.
        """
        self.name = name
        self.deterrence_type = deterrence_type
        self.deterrence_params = deterrence_params
        self.origin_exponent = origin_exponent
        self.destination_exponent = destination_exponent
        self.model_type = model_type

        try:
            self.deterrence_func = DETERRENCE_FUNCTIONS[self.deterrence_type]
        except KeyError as err:
            raise ValueError(
                f"Unsupported deterrence_type '{deterrence_type}'. "
                f"Choose from {list(DETERRENCE_FUNCTIONS)}."
            ) from err

    def _compute_gravity_score(self, location_idx: int, dist_matrix: np.ndarray, origin_relevances: np.ndarray,
                               dest_relevances: np.ndarray) -> np.ndarray:
        """
        Compute the gravity score (flow matrix) based on the gravity model formula.

        The formula used is:

        .. math::
            F_{ij} = D(d_{ij}) \cdot R_j^{\mathrm{destination\ exponent}} \cdot R_i^{\mathrm{origin\ exponent}}

        where:
            - :math:`D(d_{ij})` is the deterrence function applied to the distance between locations `i` and `j`,
            - :math:`R_i` and :math:`R_j` are the relevance scores for origin and destination locations.

        Args:
            location_idx (int): Index of the origin location in the distance matrix.
            dist_matrix (np.ndarray): Precomputed distance matrix between all locations.
            origin_relevances (np.ndarray): Relevance scores for origin locations.
            dest_relevances (np.ndarray): Relevance scores for destination locations.

        Returns:
            np.ndarray: Flow matrix of shape (n_origins, n_destinations).
        """
        deterrence = self.deterrence_func(dist_matrix, *self.deterrence_params)

        flows = (
                deterrence
                * dest_relevances[None, :] ** self.destination_exponent
                * origin_relevances[:, None] ** self.origin_exponent
        )

        # Replace nan/inf with 0
        np.nan_to_num(flows, copy=False)
        flows[~np.isfinite(flows)] = 0.0

        # Remove self‑flows
        if flows.shape[0] == flows.shape[1]:
            np.fill_diagonal(flows, 0.0)
        elif flows.shape[0] == 1:  # singly‑constrained vector
            flows[0, location_idx] = 0.0

        return flows

    compute_gravity_score = _compute_gravity_score # public alias
