from humobi.measures.individual import *
from humobi.tools.processing import *
from humobi.tools.user_statistics import *


class AnimalStatistics:
    """
    A class to calculate and store statistics related to animals
    over a given time period.

    Attributes:
    ----------
    raw_animals_no : int or None
        The number of animals before filtering.
    raw_period : pandas.Timedelta or None
        The time period before filtering.
    raw_median_meriod_for_labels : any type or None
        The median period for labels before filtering.
    filtered_animals_no : int or None
        The number of animals after filtering.
    filtered_period : pandas.Timedelta or None
        The time period after filtering.
    filtered_median_meriod_for_labels : any type or None
        The median period for labels after filtering.
    """

    def __init__(self) -> None:
        self.raw_animals_no = None
        self.raw_period = None
        self.raw_median_meriod_for_labels = None
        self.filtered_animals_no = None
        self.filtered_period = None
        self.filtered_median_meriod_for_labels = None
        self.min_records_no_before_filtration = None
        self.min_labels_no_after_filtration = None

        self.avg_duration = None
        self.min_duration = None
        self.max_duration = None

        self.overall_set_area = None
        self.average_set_area = None
        self.min_area = None
        self.max_area = None

    def get_raw_animals_no(self, trajectory_frame):
        """
        Sets the number of animals before filtering based on data
        from trajectory_frame.

        Parameters:
        ----------
        trajectory_frame : object containing trajectory data
            Trajectory data containing user information.
        """
        self.raw_animals_no = len(trajectory_frame.get_users())

    def get_raw_period(self, trajectory_frame):
        """
        Sets the time period before filtering based on data from
        trajectory_frame.

        Parameters:
        ----------
        trajectory_frame : pandas.DataFrame
            Trajectory data containing 'start' and 'end' columns
            representing the start and end times.
        """
        trajectory_frame["start"] = pd.to_datetime(trajectory_frame["start"])
        trajectory_frame["end"] = pd.to_datetime(trajectory_frame["end"])

        min_start = trajectory_frame["start"].min()
        max_end = trajectory_frame["end"].max()

        self.raw_period = max_end - min_start

    def get_raw_filtered_animals_no(self, trajectory_frame):
        """
        Sets the number of animals after filtering based on data
        from trajectory_frame.

        Parameters:
        ----------
        trajectory_frame : object containing trajectory data
            Trajectory data containing user information.
        """
        self.filtered_animals_no = len(trajectory_frame.get_users())

    def get_filtered_period(self, trajectory_frame):
        """
        Sets the time period after filtering based on data
        from trajectory_frame.

        Parameters:
        ----------
        trajectory_frame : pandas.DataFrame
            Trajectory data containing 'start' and 'end' columns
            representing the start and end times.
        """
        trajectory_frame["start"] = pd.to_datetime(trajectory_frame["start"])
        trajectory_frame["end"] = pd.to_datetime(trajectory_frame["end"])

        min_start = trajectory_frame["start"].min()
        max_end = trajectory_frame["end"].max()

        self.filtered_period = max_end - min_start


    def get_mean_periods(self, trajectory_frame):

        trajectory_frame["start"] = pd.to_datetime(trajectory_frame["start"])
        trajectory_frame["end"] = pd.to_datetime(trajectory_frame["end"])

        agg_funcs = {'start': 'min', 'end': 'max'}
        grouped = trajectory_frame.groupby('user_id').agg(agg_funcs)

        grouped['duration'] = grouped['end'] - grouped['start']
        self.avg_duration = grouped['duration'].mean()

    def get_min_periods(self, trajectory_frame):

        trajectory_frame["start"] = pd.to_datetime(trajectory_frame["start"])
        trajectory_frame["end"] = pd.to_datetime(trajectory_frame["end"])

        agg_funcs = {'start': 'min', 'end': 'max'}
        grouped = trajectory_frame.groupby('user_id').agg(agg_funcs)
        grouped['duration'] = grouped['end'] - grouped['start']
        self.min_duration = grouped['duration'].min()


    def get_max_periods(self, trajectory_frame):

        trajectory_frame["start"] = pd.to_datetime(trajectory_frame["start"])
        trajectory_frame["end"] = pd.to_datetime(trajectory_frame["end"])

        agg_funcs = {'start': 'min', 'end': 'max'}
        grouped = trajectory_frame.groupby('user_id').agg(agg_funcs)
        grouped['duration'] = grouped['end'] - grouped['start']
        self.max_duration = grouped['duration'].max()

    def get_min_records_no_before_filtration(self, raw_data):

        records_counts = raw_data.groupby('animal_id').time.count()
        min_label_count = records_counts.min()
        self.min_records_no_before_filtration = records_counts[
            records_counts == min_label_count
            ]

    def get_min_labels_no_after_filtration(self, trajectory_frame):

        unique_label_counts = trajectory_frame.groupby(
            'user_id'
            )['labels'].nunique()
        min_unique_label_count = unique_label_counts.min()
        self.min_labels_no_after_filtration = unique_label_counts[
            unique_label_counts == min_unique_label_count
            ]


    def get_overall_area(self, trajectory_frame):
        convex_hull = trajectory_frame.unary_union.convex_hull
        self.overall_set_area = round(convex_hull.area / 10000,0)


    def get_min_area(self, trajectory_frame):
        grouped = trajectory_frame.groupby('user_id')

        areas = []
        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            area_m2 = convex_hull.area
            area_ha = area_m2 / 10000
            areas.append(area_ha)
        self.min_area = round(min(areas),0)

    def get_max_area(self, trajectory_frame):
        grouped = trajectory_frame.groupby('user_id')

        areas = []
        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            area_m2 = convex_hull.area
            area_ha = area_m2 / 10000
            areas.append(area_ha)
        self.max_area = round(max(areas),0)

    def get_mean_area(self, trajectory_frame):
        grouped = trajectory_frame.groupby('user_id')
        areas = []

        for user_id, group in grouped:
            convex_hull = group.unary_union.convex_hull
            area_m2 = convex_hull.area
            area_ha = area_m2 / 10000
            areas.append(area_ha)
        self.average_set_area = round(sum(areas) / len(areas),0)


class MeasurmentsStatistics:

    def __init__(self) -> None:
        self.visitation_frequency = None
        self.distinct_locations_over_time = None
        self.jump_lengths_distribution = None
        self.waiting_times = None
        self.travel_times = None
        self.rog = None
        self.rog_over_time = None
        self.msd_distribution = None
        self.msd_curve = None
        self.return_time_distribution = None
        self.exploration_time = None

