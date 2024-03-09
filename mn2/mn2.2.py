# import numpy as np
# import matplotlib.pyplot as plt
#
# def linear_interpolation(x_known, y_known, x_interp):
#     # Perform linear interpolation
#     y_interp = np.interp(x_interp, x_known, y_known)
#
#     return y_interp
#
# # Read dataset from file
# dataset_file = 'dataset_2.txt'  # Replace with the actual file name
# data = np.loadtxt(dataset_file, delimiter=',', dtype=str)
#
# # Extract dates and visitors from the dataset
# dates = data[:, 0]
# visitors = data[:, 1].astype(int)
#
# # Convert date strings to datetime objects for plotting
# dates_datetime = np.array([np.datetime64(date) for date in dates])
#
# # Create an array of all dates in the dataset
# all_dates = np.arange(dates_datetime[0], dates_datetime[-1], dtype='datetime64[D]')
#
# # Perform linear interpolation
# visitors_interp = linear_interpolation(dates_datetime, visitors, all_dates)
#
# # Plot the original data and interpolated data
# plt.figure(figsize=(12, 6))
# plt.plot(dates_datetime, visitors, 'o-', label='Original Data')
# plt.plot(all_dates, visitors_interp, 'g--', label='Interpolated Data')
# plt.xlabel('Date')
# plt.ylabel('Number of Visitors')
# plt.title('Visitor Analysis')
# plt.legend()
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

def linear_interpolation(x_known, y_known, x_interp):
    # Perform linear interpolation
    y_interp = np.interp(x_interp, x_known, y_known)

    return y_interp

# Read the dataset from a text file
data = np.genfromtxt('dataset_2.txt', delimiter=',', dtype=str, skip_header=1)
dates = data[:, 0]
visitors = np.where(data[:, 1] == 'Nan', np.nan, data[:, 1].astype(float))

# Convert dates to numerical values for interpolation
x_known = np.arange(len(dates))
x_interp = np.arange(len(dates))

# Perform linear interpolation
y_interp = linear_interpolation(x_known[~np.isnan(visitors)], visitors[~np.isnan(visitors)], x_interp)

# Plot the original data and the interpolated data
plt.figure(figsize=(10, 6))
plt.plot(x_known, visitors, 'bo', label='Original Data')
plt.plot(x_interp, y_interp, 'r-', label='Interpolated Data')
plt.xlabel('Day')
plt.ylabel('Visitors')
plt.title('Visitor Analysis')
plt.legend()
plt.grid(True)
plt.show()

