## crs values
ELLIPSOIDAL_CRS = 4326
CARTESIAN_CRS = 3857

## infostop preprocessing/filtration params
ALLOWED_MINUTES = [1, 5, 10, 15, 20, 30, 60]
DAYS_WINDOW = 21

# level 1
RESOLUTION_OF_FRACTION_OF_MISSING_VALUES = '1H'
FRACTION_OF_MISSING_VALUES = 0.6

#level 2
RESOLUTION_OF_USER_TRAJECTORIES_DURATION = '1D'
USER_TRAJECTORIES_DURATION = 20

#level 3
TEMPORAL_RESOLUTION_EXTRACTION = '1H'

MAX_TIME_BETWEEN = 86400

#infostop params
R1_MIN = 50
R1_MAX = 10000000000
TMIN_MIN = 600
TMIN_MAX = 10000000000
BEST_POINT_NO = 15 # Take top 5 points with minimal changes and look for maximum TOTAL STOPS

## laws params filtration thresholds
MIN_LABEL_NO = 3
QUARTILE = 0.25
MIN_QUARTILE_VALUE = 3

CURVE = {'linear','expon','expon_neg'}
DISTRIBUTIONS = {'lognorm','expon','pareto','norm','truncexpon','truncnorm','truncpareto','powerlaw'}
FLEXATION_POINTS_SENSITIVITY = 'Medium'

## Pnew estimation
PNEW_P0 = [1.0, 0.5]
# reference values from literature (if applicable)
A_FIT = 0.5
B_FIT = 0.825
GAMMA = 0.6
RHO = 0.21
