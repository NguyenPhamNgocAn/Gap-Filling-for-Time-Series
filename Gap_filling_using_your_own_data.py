import pandas as pd #library for dataframe processing
import numpy as np # array processing
import matplotlib.pyplot as plt # visualization, plotting
import datetime # datetime processing
from scipy import interpolate # interpolation package

### THIS FILE IS FOR FILLING MISSING VALUES IN YOUR OWN DATA

your_file_name = "your_data.csv"
column = "temp"
df = pd.read_csv(your_file_name)

#Try to assign x, x_missing, y, y_true here

### get indexes where the data are missing or not
nan_offsets = df[column].isna().values
nonnan_offsets = ~nan_offsets

# assuming the data is the right order, x will be an interger array from 1 to lenth of the data
x = np.arange(len(df))[nonnan_offsets]  # non-missing offsets
y = df[column].values[nonnan_offsets]   # non-missing values
x_missing = np.arange(len(df))[nan_offsets]


# 1D Nearest Interpolation

# do interpolation
interpolate_function = interpolate.interp1d(x, y, kind='nearest')
y_interpolated = interpolate_function(x_missing)

# assign back to the dataframe, here we create a new column as interpolated column
interpolated_column_name = 'interp1d.nearest'
df[interpolated_column_name] = df[column]
df.loc[nan_offsets, interpolated_column_name] = y_interpolated


# 1D Linear Interpolation

# do interpolation
interpolate_function = interpolate.interp1d(x, y, kind='linear')
y_interpolated = interpolate_function(x_missing)

# assign back to the dataframe, here we create a new column as interpolated column
interpolated_column_name = 'interp1d.linear'
df[interpolated_column_name] = df[column]
df.loc[nan_offsets, interpolated_column_name] = y_interpolated


# 1D Spline Interpolation

# LOOP over k (the polynomial degree)

for k in [2,3,5,7,9]:
    # do interpolation
    interpolate_function = interpolate.make_interp_spline(x, y, k=k)
    y_interpolated = interpolate_function(x_missing)

    # assigin back to the dataframe, here we create a new column as interpolated column
    interpolated_column_name = f"make_interp_spline.{k}"
    df[interpolated_column_name] = df['temp']
    df.loc[nan_offsets, interpolated_column_name] = y_interpolated


# 1D Akima Interpolator Interpolation

interpolate_function = interpolate.Akima1DInterpolator(x, y)
y_interpolated = interpolate_function(x_missing)

# assigin back to the dataframe, here we create a new column as interpolated column
interpolated_column_name = 'Akima1DInterpolator'
df[interpolated_column_name] = df['temp']
df.loc[nan_offsets, interpolated_column_name] = y_interpolated


# 1D PchipInterpolator Interpolation

interpolate_function = interpolate.PchipInterpolator(x, y)
y_interpolated = interpolate_function(x_missing)

# assigin back to the dataframe, here we create a new column as interpolated column
interpolated_column_name = 'PchipInterpolator'
df[interpolated_column_name] = df['temp']
df.loc[nan_offsets, interpolated_column_name] = y_interpolated

