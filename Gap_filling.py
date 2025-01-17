import pandas as pd #library for dataframe processing
import numpy as np # array processing
import matplotlib.pyplot as plt # visualization, plotting
import datetime # datetime processing
from scipy import interpolate # interpolation package

# Import data
df_station = pd.read_csv("gapfilling/station.csv")

# Original data

# FILE: gapfilling/original_{station_name}.csv has 5 columns
#
# datetime: date and time the temperature is recorded
#
# station: name of the station
#
# station_lat: latitude of the station
#
# station_long: longitude of the station
#
# temp: the temperature at the station at the time
#
#
# These are original data, NO NaN values on the temp column.
df_original_station = pd.read_csv("gapfilling/original_s2-2.csv")

df_gap_station = pd.read_csv("gapfilling/gap_s2-2.csv")

## 1-D Interpolation
# Before doing the gapfilling, we need to process the data into appropriate datatypes.
# Such as convert datetime to integers as follow

# Load data
df_gap_station = pd.read_csv("gapfilling/gap_s2-2.csv")

# Converte 'datetime' column to datetime format
df_gap_station['datetime'] = df_gap_station['datetime'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

# Convert and normalize datetime to integers
df_gap_station['time_int'] = df_gap_station['datetime'].astype(int)
df_gap_station['time_int'] = (df_gap_station['time_int'] - df_gap_station['time_int'].min())/ 3600000000000
df_gap_station['time_int'] = df_gap_station['time_int'].astype(int)

# remove unesscessary columns
df_gap_station = df_gap_station[['time_int', 'temp']]

# Load original data (the rows are already aligned here, otherwise aligning rows across dataframes such as joining dataframes must be done)
df_original_station = pd.read_csv("gapfilling/original_s2-2.csv")

# Try to assign x, x_missing, y, y_true here

# get indexes where the data are missing or not
nan_offsets = df_gap_station['temp'].isna().values
nonnan_offsets = ~nan_offsets

x = df_gap_station['time_int'].values[nonnan_offsets]  ## non-missing time
y = df_gap_station['temp'].values[nonnan_offsets]   # non-missing temp
x_missing = df_gap_station['time_int'].values[nan_offsets] # time where temp is missing
y_true = df_original_station['temp'].values[nan_offsets]   # groundtruth for temp (for those missing)

# Nearest Interpolation
# do interpolation
interpolate_function = interpolate.interp1d(x, y, kind='nearest')
y_interpolated = interpolate_function(x_missing)

# assigin back to the dataframe, here we create a new column as interpolated column
interpolated_column_name = 'interp1d.nearest'
df_gap_station[interpolated_column_name] = df_gap_station['temp']
df_gap_station.loc[nan_offsets, interpolated_column_name] = y_interpolated

# calculate RMSE
rmse = np.sqrt(np.mean(np.square(y_true - df_gap_station[interpolated_column_name].values[nan_offsets])))
print("RMSE: ", rmse)

# Linear Interpolation

# do interpolation
interpolate_function = interpolate.interp1d(x, y, kind='linear')
y_interpolated = interpolate_function(x_missing)

# assigin back to the dataframe, here we create a new column as interpolated column
interpolated_column_name = 'interp1d.linear'
df_gap_station[interpolated_column_name] = df_gap_station['temp']
df_gap_station.loc[nan_offsets, interpolated_column_name] = y_interpolated

# calculate RMSE
rmse = np.sqrt(np.mean(np.square(y_true - df_gap_station[interpolated_column_name].values[nan_offsets])))
print("RMSE: ", rmse)

# Spline

## LOOP over k (the polynomial degree)

for k in [2,3,5,7,9]:
    # do interpolation
    interpolate_function = interpolate.make_interp_spline(x, y, k=k)
    y_interpolated = interpolate_function(x_missing)

    # assigin back to the dataframe, here we create a new column as interpolated column
    interpolated_column_name = f"make_interp_spline.{k}"
    df_gap_station[interpolated_column_name] = df_gap_station['temp']
    df_gap_station.loc[nan_offsets, interpolated_column_name] = y_interpolated

    # calculate RMSE
    rmse = np.sqrt(np.mean(np.square(y_true - df_gap_station[interpolated_column_name].values[nan_offsets])))
    print("K = ", "RMSE: ", rmse)



# Akima

# do interpolation
interpolate_function = interpolate.Akima1DInterpolator(x, y)
y_interpolated = interpolate_function(x_missing)

# assigin back to the dataframe, here we create a new column as interpolated column
interpolated_column_name = 'Akima1DInterpolator'
df_gap_station[interpolated_column_name] = df_gap_station['temp']
df_gap_station.loc[nan_offsets, interpolated_column_name] = y_interpolated

# calculate RMSE
rmse = np.sqrt(np.mean(np.square(y_true - df_gap_station[interpolated_column_name].values[nan_offsets])))
print("RMSE: ", rmse)

# Pchip

# do interpolation
interpolate_function = interpolate.PchipInterpolator(x, y)
y_interpolated = interpolate_function(x_missing)

# assigin back to the dataframe, here we create a new column as interpolated column
interpolated_column_name = 'PchipInterpolator'
df_gap_station[interpolated_column_name] = df_gap_station['temp']
df_gap_station.loc[nan_offsets, interpolated_column_name] = y_interpolated

# calculate RMSE
rmse = np.sqrt(np.mean(np.square(y_true - df_gap_station[interpolated_column_name].values[nan_offsets])))
print("RMSE: ", rmse)

## Multi-dimensional Interpolation
# As an illustration, we use time, latitude and longitude as 3 space-time variables to interpolate  missing temperature values.
# Preprocessing
# Input variables need to be converted to float and normalized to better scales before interpolation.

# load files, concat the files
df_gap = pd.concat([pd.read_csv(f"gapfilling/gap_{station}.csv") for station in df_station.station.values])

# Converte 'datetime' column to datetime format
df_gap['datetime'] = df_gap['datetime'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

# Convert and normalize datetime to integers
df_gap['time_int'] = df_gap['datetime'].astype(int)
df_gap['time_int'] = (df_gap['time_int'] - df_gap['time_int'].min())/ 3600000000000
df_gap['time_int'] = df_gap['time_int'].astype(int)

# trying to normalize latitude and longitude in the df_station dataframe
# first convert latitude and longitude to Cartesian cooridnates by harversine fomular
from math import radians, sin, cos, sqrt, atan2
def haversine(coord1, coord2):
    # calculate the distance between two coordinates
    R = 6371.0  # radius of the Earth in kilometers

    lat1, lon1 = map(radians, coord1)
    lat2, lon2 = map(radians, coord2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def _transform_coordindates(latlng, origin):
    # transform to Cartesian coordinates by calculate the distance to the origin
    x1 = haversine((latlng[0], origin[1]), (origin[0], origin[1]))
    if latlng[0] < origin[0]:
        x1 = -x1
    x2 = haversine((origin[0], latlng[1]), (origin[0], origin[1]))
    if latlng[1] < origin[1]:
        x2 = -x2
    return x1, x2

coordinate_origin = ((df_station.station_lat.min() + df_station.station_lat.max())/2,
              (df_station.station_long.min() + df_station.station_long.max())/2)
transform_coordindates = lambda x: _transform_coordindates(x, coordinate_origin)
x1, x2 = list(zip(*[transform_coordindates((lat,lng)) for lat, lng in df_station[["station_lat", "station_long"]].values]))
df_station['x1'] = np.asarray(x1) * 0.5 # rescale
df_station['x2'] = np.asarray(x2) * 0.5

df_gap = df_gap.join(df_station[["station", "x1", "x2"]].set_index('station'), on='station')

# keep neccessary columns only
df_gap = df_gap[['time_int', 'x1', 'x2', 'temp']]

# sort by time, x1, x2
df_gap.sort_values(by=['time_int', 'x1', 'x2'], inplace=True)

# similarly we do the same for original data
df_original = pd.concat([pd.read_csv(f"gapfilling/original_{station}.csv") for station in df_station.station.values])

df_original['datetime'] = df_original['datetime'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

df_original['time_int'] = df_original['datetime'].astype(int)
df_original['time_int'] = (df_original['time_int'] - df_original['time_int'].min())/ 3600000000000
df_original['time_int'] = df_original['time_int'].astype(int)

df_original = df_original.join(df_station[["station", "x1", "x2"]].set_index('station'), on='station')

df_original = df_original[['time_int', 'x1', 'x2', 'temp']]

df_original.sort_values(by=['time_int', 'x1', 'x2'], inplace=True) # align with the df_gap now

# Nearest Interpolation

#Try to assign x, x_missing, y, y_true here

### get indexes where the data are missing or not
nan_offsets = df_gap['temp'].isna().values
nonnan_offsets = ~nan_offsets

x = df_gap[['time_int', 'x1', 'x2']].values[nonnan_offsets]  ## non-missing row
y = df_gap['temp'].values[nonnan_offsets]   # non-missing temp
x_missing = df_gap[['time_int', 'x1', 'x2']].values[nan_offsets] # time & location where value is missing
y_true = df_original['temp'].values[nan_offsets]   # groundtruth for temp (for those missing)

# do interpolation
interpolate_function = interpolate.NearestNDInterpolator(x, y)
y_interpolated = interpolate_function(x_missing)

# assigin back to the dataframe, here we create a new column as interpolated column
interpolated_column_name = 'NearestNDInterpolator'
df_gap[interpolated_column_name] = df_gap['temp']
df_gap.loc[nan_offsets, interpolated_column_name] = y_interpolated

# calculate RMSE
rmse = np.sqrt(np.mean(np.square(y_true - df_gap[interpolated_column_name].values[nan_offsets])))
print("RMSE: ", rmse)

# Linear Interpolation
# Due to the computational complexity in high dimensionality, data is fitted and interpolated within smaller windows
# rather than across the whole dataset.

WINDOW_SIZE = 20000
PADDING_SIZE =1000        # PADDING at the boundary

cur_df_gap = df_gap[['time_int', 'x1', 'x2', 'temp']].copy() # make a copy to run this method
interpolated_column_name = 'NearestNDInterpolator'
cur_df_gap[interpolated_column_name] = df_gap['temp']
df_output = []
for ii in range(0, len(df_gap), WINDOW_SIZE):
    # slice df window
    cur_df_gap_window = cur_df_gap.iloc[max(0,ii-PADDING_SIZE):ii+WINDOW_SIZE+PADDING_SIZE].reset_index(drop=True)

    # assign x, y, x_misisng fur current window
    nonnan_offsets = np.where(~cur_df_gap_window.temp.isna())[0]
    nan_offsets = np.where(cur_df_gap_window.temp.isna())[0]
    x = cur_df_gap_window[['time_int', 'x1', 'x2']].values[nonnan_offsets]
    y = cur_df_gap_window['temp'].values[nonnan_offsets]
    x_missing = cur_df_gap_window[['time_int', 'x1', 'x2']].values[nan_offsets]

    ## do interpolation
    interpolate_function = interpolate.LinearNDInterpolator(x, y)
    y_interpolated = interpolate_function(x_missing)

    # assigin back to the current dataframe
    cur_df_gap_window.loc[nan_offsets, interpolated_column_name] = y_interpolated

    # save window dfs, and then concat them later
    start_index = PADDING_SIZE if ii > PADDING_SIZE else 0
    df_output.append(cur_df_gap_window.iloc[start_index:start_index+WINDOW_SIZE])   # cut out the padding
df_output = pd.concat(df_output)
df_gap[interpolated_column_name] = df_output[interpolated_column_name].values

# calculate RMSE
nan_offsets = df_gap['temp'].isna().values
rmse = np.sqrt(np.nanmean(np.square(df_original["temp"].values[nan_offsets] - df_gap[interpolated_column_name].values[nan_offsets])))
print("RMSE: ", rmse)


# RBF Interpolation
# Due to the computational complexity in high dimensionality, data is fitted and interpolated within smaller windows
# rather than across the whole dataset. It may take some time.

WINDOW_SIZE = 10000
PADDING_SIZE =1000        # PADDING at the boundary

for kernel in ['linear', 'thin_plate_spline', 'cubic', 'gaussian']:
    cur_df_gap = df_gap[['time_int', 'x1', 'x2', 'temp']].copy() # make a copy to run this method
    interpolated_column_name = 'NearestNDInterpolator'
    cur_df_gap[interpolated_column_name] = df_gap['temp']
    df_output = []
    for ii in range(0, len(df_gap), WINDOW_SIZE):
        # slice df window
        cur_df_gap_window = cur_df_gap.iloc[max(0,ii-PADDING_SIZE):ii+WINDOW_SIZE+PADDING_SIZE].reset_index(drop=True)

        # assign x, y, x_misisng fur current window
        nonnan_offsets = np.where(~cur_df_gap_window.temp.isna())[0]
        nan_offsets = np.where(cur_df_gap_window.temp.isna())[0]
        x = cur_df_gap_window[['time_int', 'x1', 'x2']].values[nonnan_offsets]
        y = cur_df_gap_window['temp'].values[nonnan_offsets]
        x_missing = cur_df_gap_window[['time_int', 'x1', 'x2']].values[nan_offsets]

        ## do interpolation
        interpolate_function = interpolate.RBFInterpolator(x, y, kernel=kernel, epsilon=1)
        y_interpolated = interpolate_function(x_missing)

        # assigin back to the current dataframe
        cur_df_gap_window.loc[nan_offsets, interpolated_column_name] = y_interpolated

        # save window dfs, and then concat them later
        start_index = PADDING_SIZE if ii > PADDING_SIZE else 0
        df_output.append(cur_df_gap_window.iloc[start_index:start_index+WINDOW_SIZE])   # cut out the padding
    df_output = pd.concat(df_output)
    df_gap[interpolated_column_name] = df_output[interpolated_column_name].values

    # calculate RMSE
    nan_offsets = df_gap['temp'].isna().values
    rmse = np.sqrt(np.nanmean(np.square(df_original["temp"].values[nan_offsets] - df_gap[interpolated_column_name].values[nan_offsets])))
    print("Kernel: ", kernel, ", RMSE: ", rmse)



