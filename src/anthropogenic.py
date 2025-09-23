import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from geopandas import GeoDataFrame
from typing import Optional, Union
import seaborn as sns
from scipy.stats import pearsonr


def calculate_roads_intersections(
    animal_data: GeoDataFrame,
    roads_data: GeoDataFrame,
    roads_type: Union[str, int],
    animals_concave_hulls: GeoDataFrame
) -> None:
    """
    Calculate and visualize intersections between animal ranges and a specific type of road.

    Args:
        animal_data (GeoDataFrame): Spatial data of animals, must contain 'user_id'.
        roads_data (GeoDataFrame): Spatial data of roads, must contain 'KAT_ZARZAD'.
        roads_type (str | int): The type of road to analyze.
        animals_concave_hulls (GeoDataFrame): Precomputed concave hulls of animal ranges.

    Returns:
        None
    """
    roads = roads_data[roads_data["KAT_ZARZAD"] == roads_type]

    hulls_with_roads = gpd.sjoin(animal_data, roads, predicate="intersects")
    hulls_intersect = animal_data[
        animal_data["user_id"].isin(hulls_with_roads["user_id"])
    ]

    ratio = round(len(hulls_intersect) / len(animal_data), 3)
    print(f"Intersection ratio: {ratio}%")
    print(f"Animals with intersections: {hulls_intersect['user_id'].unique()}")

    ax = animals_concave_hulls.boundary.plot(color="gray", figsize=(8, 6))
    roads.plot(ax=ax, color="orange")
    hulls_intersect.boundary.plot(ax=ax, color="red", linewidth=2)

    plt.title(f"Roads ({roads_type}) Intersections with Animal Ranges")
    plt.show()

    hulls_intersect.to_csv(f"{roads_type}.csv", index=False)


def calculate_rivers_intersections(
    animal_data: GeoDataFrame,
    rivers_data: GeoDataFrame,
    river_type: Union[str, int],
    animals_concave_hulls: GeoDataFrame
) -> None:
    """
    Calculate and visualize intersections between animal ranges and a specific type of river.

    Args:
        animal_data (GeoDataFrame): Spatial data of animals, must contain 'user_id'.
        rivers_data (GeoDataFrame): Spatial data of rivers, must contain 'RODZAJ'.
        river_type (str | int): The type of river to analyze.
        animals_concave_hulls (GeoDataFrame): Precomputed concave hulls of animal ranges.

    Returns:
        None
    """
    rivers = rivers_data[rivers_data["RODZAJ"] == river_type]

    hulls_with_rivers = gpd.sjoin(animal_data, rivers, predicate="intersects")
    hulls_intersect = animal_data[
        animal_data["user_id"].isin(hulls_with_rivers["user_id"])
    ]

    ratio = round(len(hulls_intersect) / len(animal_data), 3)
    print(f"Intersection ratio: {ratio}%")
    print(f"Animals with intersections: {hulls_intersect['user_id'].unique()}")

    if not hulls_intersect.empty:
        ax = animals_concave_hulls.boundary.plot(color="gray", figsize=(8, 6))
        rivers.plot(ax=ax, color="blue")
        hulls_intersect.boundary.plot(ax=ax, color="red", linewidth=2)

        plt.title(f"Rivers ({river_type}) Intersections with Animal Ranges")
        plt.show()

        hulls_intersect.to_csv(f"{river_type}.csv", index=False)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from typing import Union


def analyze_hunting_areas(
    hunting_area: GeoDataFrame,
    sorted_animals: GeoDataFrame,
    hunting_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze animal movement data in relation to hunting periods within defined hunting areas.

    For each hunting area:
      - Plots hunting periods against animal presence dates.
      - Separates animal movement into inside and outside hunting periods.
      - Calculates mean and standard deviation of movement distances for both cases.
      - Collects results into a summary DataFrame.

    Args:
        hunting_area (GeoDataFrame): Polygons of hunting areas with a 'NAME' column.
        sorted_animals (GeoDataFrame): Animal tracking points, must contain 'datetime' and 'dist_to_prev_km'.
        hunting_data (pd.DataFrame): Hunting period records with columns:
                                     'Dzialki', 'Numer pozwolenia',
                                     'Data rozpoczęcia', 'Data zakończenia'.

    Returns:
        pd.DataFrame: Summary results with columns:
                      ['area', 'mean_distance_in_hunting', 'std_distance_in_hunting',
                       'mean_distance_out_hunting', 'std_distance_out_hunting'].
    """
    hunting_results = pd.DataFrame(
        columns=[
            "area",
            "mean_distance_in_hunting",
            "std_distance_in_hunting",
            "mean_distance_out_hunting",
            "std_distance_out_hunting"
        ]
    )

    def in_period(ts: pd.Timestamp, periods: pd.DataFrame) -> bool:
        """Check if a timestamp falls inside any hunting period."""
        return any(
            (ts >= row["Data rozpoczęcia"]) & (ts <= row["Data zakończenia"])
            for _, row in periods.iterrows()
        )

    def out_period(ts: pd.Timestamp, periods: pd.DataFrame) -> bool:
        """Check if a timestamp falls outside all hunting periods."""
        return not in_period(ts, periods)

    for name in hunting_area["NAME"].unique():
        print(f"Processing hunting area: {name}")
        polygon = hunting_area[hunting_area["NAME"] == name].geometry.unary_union

        # Select animal points inside the hunting area
        pts_in_area = sorted_animals[sorted_animals.within(polygon)]
        periods_area = hunting_data[hunting_data["Dzialki"] == name]

        print(f"Number of records in area: {len(pts_in_area)}")

        if len(pts_in_area) > 0:
            # Plot hunting periods and animal presence
            fig, ax = plt.subplots(figsize=(10, 6))

            for _, row in periods_area.iterrows():
                ax.barh(
                    y=row["Numer pozwolenia"],
                    width=(row["Data zakończenia"] - row["Data rozpoczęcia"]).days,
                    left=row["Data rozpoczęcia"],
                    height=0.5,
                    color="skyblue"
                )

            ax.axvline(
                pts_in_area["datetime"].min(),
                color="green", linestyle="--",
                label="First animal presence"
            )
            ax.axvline(
                pts_in_area["datetime"].max(),
                color="red", linestyle="--",
                label="Last animal presence"
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Permit number")
            ax.set_title(f"Hunting periods for area {name}")
            plt.tight_layout()
            plt.legend()
            plt.show()

            # Split animal points by hunting period
            pts_inside = pts_in_area[
                pts_in_area["datetime"].apply(lambda x: in_period(x, periods_area))
            ]
            pts_outside = pts_in_area[
                pts_in_area["datetime"].apply(lambda x: out_period(x, periods_area))
            ]

            print(f"Records inside hunting period: {len(pts_inside)}")
            print(f"Records outside hunting period: {len(pts_outside)}")

            # Compute statistics
            mean_in = pts_inside["dist_to_prev_km"].mean()
            std_in = pts_inside["dist_to_prev_km"].std()
            mean_out = pts_outside["dist_to_prev_km"].mean()
            std_out = pts_outside["dist_to_prev_km"].std()

            print(f"Mean movement inside hunting: {mean_in}, std: {std_in}")
            print(f"Mean movement outside hunting: {mean_out}, std: {std_out}")

            # Append results
            hunting_results = pd.concat(
                [
                    hunting_results,
                    pd.DataFrame({
                        "area": [name],
                        "mean_distance_in_hunting": [mean_in],
                        "std_distance_in_hunting": [std_in],
                        "mean_distance_out_hunting": [mean_out],
                        "std_distance_out_hunting": [std_out]
                    })
                ],
                ignore_index=True
            )

    return hunting_results


def analyze_landcover_distribution(
    landcover_path: str,
    animals_data: GeoDataFrame,
    animals_concave_hulls: GeoDataFrame,
    target_crs: int = 2180
) -> pd.DataFrame:
    """
    Analyze the distribution of animal points across different landcover classes.

    Steps performed:
      1. Load and reproject landcover shapefile.
      2. Combine animal concave hulls into a single polygon.
      3. Select landcover polygons that contain animal points.
      4. Clip selected polygons to the combined animal range.
      5. Count animal points within each landcover class.
      6. Compute area (after reprojecting to target CRS).
      7. Calculate density and normalized densities for each class.

    Args:
        landcover_path (str): Path to the landcover shapefile (.shp).
        animals_data (GeoDataFrame): Animal point data with a geometry column.
        animals_concave_hulls (GeoDataFrame): Concave hulls of animal ranges.
        target_crs (int, optional): CRS (EPSG code) used for area calculation. Default = 2180.

    Returns:
        pd.DataFrame: Summary statistics by landcover class with columns:
                      ['RODZAJ', 'count', 'area', 'density', 'normalized', 'percent_normalized'].
    """
    # Load landcover and reproject
    lc_data = gpd.read_file(landcover_path).to_crs(4326)

    # Merge concave hulls into one polygon
    combined_polygon = unary_union(animals_concave_hulls.geometry)
    combined_gdf = gpd.GeoDataFrame(geometry=[combined_polygon], crs=animals_concave_hulls.crs)

    # Find landcover polygons containing animal points
    joined = gpd.sjoin(lc_data, animals_data, how="inner", predicate="contains")
    polygons_with_points = lc_data.loc[lc_data.index.isin(joined.index)]

    # Clip polygons to combined animal range
    clipped_polygons = gpd.overlay(polygons_with_points, combined_gdf, how="intersection")

    # Join animal points with clipped landcover polygons
    joined = gpd.sjoin(animals_data, clipped_polygons[["RODZAJ", "geometry"]], predicate="within")

    # Count points per landcover class
    counts = joined.groupby("RODZAJ").size().reset_index(name="count")

    # Calculate polygon areas in target CRS
    lc_data_m = clipped_polygons.to_crs(epsg=target_crs)
    lc_area = lc_data_m.dissolve(by="RODZAJ")
    lc_area["area"] = lc_area.geometry.area

    # Merge counts with area
    counts = counts.merge(lc_area[["area"]], left_on="RODZAJ", right_index=True)

    # Density and normalization
    counts["density"] = counts["count"] / counts["area"]
    total_points = counts["count"].sum()
    counts["normalized"] = counts["density"] / total_points
    counts["percent_normalized"] = 100 * counts["normalized"]

    return counts


def load_and_resample(filepath: str, freq: str = "15T") -> GeoDataFrame:
    """
    Load animal GPS data from CSV, set datetime index, and resample the time series.

    Args:
        filepath (str): Path to the CSV file. Must contain ['lon', 'lat', 'datetime', 'user_id'].
        freq (str): Resampling frequency, default "15T" (15 minutes).

    Returns:
        GeoDataFrame: Resampled animal data with geometry column.
    """
    df = pd.read_csv(filepath, index_col=0)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").sort_index()

    def resample_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g[~g.index.duplicated(keep="first")]
        return g.resample(freq).ffill()

    resampled = (
        df.groupby("user_id", group_keys=False)
          .apply(resample_group)
          .reset_index()
    )
    resampled["geometry"] = [Point(xy) for xy in zip(resampled["lon"], resampled["lat"])]
    return gpd.GeoDataFrame(resampled, geometry="geometry", crs="EPSG:4326")


def calculate_landcover_usage(animals_gdf: GeoDataFrame, landcover_path: str) -> pd.DataFrame:
    """
    Calculate landcover availability, usage and selection ratio.

    Args:
        animals_gdf (GeoDataFrame): Animal points with 'RODZAJ' column (landcover type).
        landcover_path (str): Path to landcover shapefile.

    Returns:
        pd.DataFrame: Summary table with used_ratio, available_ratio, selection_ratio.
    """
    landcover = gpd.read_file(landcover_path).to_crs(3857)
    landcover["area"] = landcover.geometry.area
    total_area = landcover["area"].sum()

    area_ratios = (
        landcover.groupby("RODZAJ")["area"].sum() / total_area
    ).reset_index()
    area_ratios.columns = ["RODZAJ", "available_ratio"]

    use_counts = animals_gdf.groupby("RODZAJ").size().reset_index(name="used")
    use_counts["used_ratio"] = use_counts["used"] / use_counts["used"].sum()

    comp = pd.merge(use_counts, area_ratios, on="RODZAJ", how="outer").fillna(0)
    comp["selection_ratio"] = comp["used_ratio"] / comp["available_ratio"]
    return comp


def calculate_distance_to_roads(
    animals_gdf: GeoDataFrame, roads_path: str, road_type: str
) -> GeoDataFrame:
    """
    Calculate distance from animal points to nearest road of given type.

    Args:
        animals_gdf (GeoDataFrame): Animal points.
        roads_path (str): Path to roads shapefile.
        road_type (str): Road type to filter, e.g. 'powiatowa', 'gminna'.

    Returns:
        GeoDataFrame: Animal points with new column 'dist_to_road'.
    """
    roads = gpd.read_file(roads_path)
    animals_gdf = animals_gdf.to_crs(roads.crs)
    roads_sel = roads[roads["KAT_ZARZAD"] == road_type]

    animals_gdf["dist_to_road"] = animals_gdf.geometry.apply(
        lambda p: roads_sel.distance(p).min()
    )
    return animals_gdf


def detect_road_crossings(
    animals_gdf: GeoDataFrame, roads_gdf: GeoDataFrame, buffer_dist: float = 20
) -> GeoDataFrame:
    """
    Detect road crossings by creating trajectories and checking intersections.

    Args:
        animals_gdf (GeoDataFrame): Animal GPS points with columns 'user_id', 'datetime', 'geometry'.
        roads_gdf (GeoDataFrame): Road geometries.
        buffer_dist (float): Buffer distance around roads to detect crossings (default = 20m).

    Returns:
        GeoDataFrame: Subset of trajectories crossing buffered roads.
    """
    gdf = animals_gdf.to_crs(roads_gdf.crs).sort_values(["user_id", "datetime"])

    # Build trajectories
    lines = []
    for uid, group in gdf.groupby("user_id"):
        group = group.sort_values("datetime")
        for p1, p2 in zip(group.geometry[:-1], group.geometry[1:]):
            if p1 != p2:
                lines.append({"user_id": uid, "geometry": LineString([p1, p2])})

    traj = gpd.GeoDataFrame(lines, crs=gdf.crs)
    road_buf = roads_gdf.buffer(buffer_dist)
    crossings = traj[traj.intersects(road_buf.unary_union)]
    return crossings


def plot_animals_roads(
    animals_gdf: GeoDataFrame,
    traj: Optional[GeoDataFrame],
    roads_gdf: GeoDataFrame,
    title: str = "Animals, Roads, and Trajectories"
) -> None:
    """
    Plot animals, roads, and trajectories.

    Args:
        animals_gdf (GeoDataFrame): Animal points.
        traj (GeoDataFrame | None): Trajectories (optional).
        roads_gdf (GeoDataFrame): Roads.
        title (str): Plot title.

    Returns:
        None
    """
    ax = roads_gdf.plot(color="black", linewidth=1, figsize=(10, 10))
    if traj is not None:
        traj.plot(ax=ax, color="blue", alpha=0.3, linewidth=0.8, label="trajectories")
    animals_gdf.plot(ax=ax, color="orange", markersize=5, label="animals")
    plt.legend()
    plt.title(title)
    plt.show()

def haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Calculate the Haversine distance (in meters) between two sets of coordinates."""
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def analyze_boars_vs_mushrooms(boars_file: str, mushrooms_file: str, threshold: int = 50) -> None:
    """Full pipeline: load data, compute boar trajectories, analyze correlation with mushroom presence,
    plot results, and compare groups.

    Args:
        boars_file (str): Path to the CSV file containing boars' trajectories.
        mushrooms_file (str): Path to the CSV file containing mushroom pickers' presence data.
        threshold (int, optional): Threshold for high vs low pressure groups. Defaults to 50.
    """
    # Load and preprocess data
    boars = pd.read_csv(boars_file)
    grzyby = pd.read_csv(mushrooms_file)

    boars['datetime'] = pd.to_datetime(boars['datetime'])
    boars['date'] = boars['datetime'].dt.date
    grzyby['date'] = pd.to_datetime(grzyby['date'], dayfirst=True).dt.date

    # Compute daily trajectories
    boars = boars.sort_values(['user_id', 'datetime']).reset_index(drop=True)
    boars['lat_shift'] = boars.groupby('user_id')['lat'].shift()
    boars['lon_shift'] = boars.groupby('user_id')['lon'].shift()

    boars['step_m'] = haversine(
        boars['lat_shift'], boars['lon_shift'], boars['lat'], boars['lon']
    )

    traj_lengths = (
        boars.groupby(['user_id', 'date'])['step_m']
        .sum()
        .reset_index(name='trajectory_m')
    )

    # Merge with mushroom presence data
    merged = traj_lengths.merge(grzyby, on='date', how='left')

    # Correlation analysis
    corr, pval = pearsonr(merged['presence'], merged['trajectory_m'])
    print(f"Pearson correlation r = {corr:.3f}, p = {pval:.3f}")

    # Plot relationship
    plt.figure(figsize=(8, 5))
    sns.regplot(data=merged, x='presence', y='trajectory_m', scatter_kws={'alpha': 0.5})
    plt.xlabel("Number of mushroom pickers (presence)")
    plt.ylabel("Daily boar trajectory length [m]")
    plt.title("Impact of mushroom pickers on boar movement")
    plt.show()

    # Compare groups
    merged['pressure_group'] = merged['presence'].apply(
        lambda x: "High" if x >= threshold else "Low"
    )

    group_means = merged.groupby('pressure_group')['trajectory_m'].mean()
    print("Average daily trajectory lengths [m]:")
    print(group_means)