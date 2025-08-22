"""
Script for annotating animal trajectory data with interpolated weather
parameters and coverage data.

This script uses two components:
1. WeatherCoverage - performs IDW interpolation of weather station
   parameters and attaches them to animal trajectory points.
2. CoverageSampler - samples raster-based coverage data for the same
   trajectory points.

Inputs:
    ANIMAL_DATA (str): Path to the animal trajectories CSV.
    WEATHER_DATA (str): Path to the weather station data CSV.
    COVERAGE_DATA (str): Path to the coverage raster file.

Outputs:
    - "animal_with_weather_data_annotation.csv"
    - "animal_with_coverage_data_annotation.csv"
"""

from src.coverage import CoverageSampler, WeatherCoverage

ANIMAL_DATA = ''
WEATHER_DATA = ''
COVERAGE_DATA = ''


def main():
    """
    Main pipeline for enriching animal trajectory data with weather and coverage information.

    Workflow:
        1. Load animal trajectories and weather data.
        2. Perform IDW interpolation of weather parameters for each trajectory point.
        3. Save results to "animal_with_weather_data_annotation.csv".
        4. Load coverage raster and animal trajectories.
        5. Sample coverage data for each trajectory point.
        6. Save results to "animal_with_coverage_data_annotation.csv".

    Returns:
        None
    """
    weather_cov = WeatherCoverage(ANIMAL_DATA, WEATHER_DATA)
    weather_coverage = weather_cov.interpolate_by_idw()
    weather_coverage.to_csv('animal_with_weather_data_annotation.csv')

    cs = CoverageSampler()
    cs.read_raster(COVERAGE_DATA, crs='2180')
    cs.read_trajectiories(ANIMAL_DATA)
    results = cs.get_coverage()

    results.to_csv('animal_with_coverage_data_annotation.csv')

if __name__ == "__main__":
    # Ensure the main function is called when the script is executed directly.
    main()
