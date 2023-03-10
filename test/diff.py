# import numpy as np
# import peakutils
#
# # Define the data as a numpy array
# data = np.array([1.384131, 1.490517, 1.553492, 1.579048, 1.594128, 1.602075, 1.755714, 1.925522,
#                  1.935053, 1.953043, 2.168038, 2.207686, 2.289834, 2.303781, 2.314795, 2.317273,
#                  2.322956, 2.326025, 2.328032, 2.346886, 2.354793, 2.412046, 2.436455, 2.484988,
#                  2.53577, 2.606285, 2.638595, 2.682385, 2.69403, 2.873687, 3.022308, 3.230573,
#                  3.638696, 3.771732, 3.991832, 4.059629, 4.203285, 4.475985, 4.615193, 5.033052,
#                  5.04564, 5.073794, 5.179396, 6.160659, 6.601993, 6.617971, 8.257484, 8.427954,
#                  8.783522, 10.829314])
#
# # Detect peaks using the indexes function with adaptive thresholds
# threshold = 0.1 * np.mean(data)
# peaks = peakutils.indexes(data, thres=threshold, min_dist=1)
#
# # Check if any peaks were found
# if len(peaks) == 0:
#     print("No significant peaks found.")
# else:
#     # Get the index and value of the peak closest to the target value
#     target_value = 6.160659
#     peak_index = np.argmin(np.abs(data[peaks] - target_value))
#     peak_value = data[peaks][peak_index]
#
#     # Print the index and value of the peak
#     print("Peak index: ", peaks[peak_index])
#     print("Peak value: ", peak_value)

import numpy as np

# Define the data as a numpy array
data = np.array([1.384131, 1.490517, 1.553492, 1.579048, 1.594128, 1.602075, 1.755714, 1.925522,
                 1.935053, 1.953043, 2.168038, 2.207686, 2.289834, 2.303781, 2.314795, 2.317273,
                 2.322956, 2.326025, 2.328032, 2.346886, 2.354793, 2.412046, 2.436455, 2.484988,
                 2.53577, 2.606285, 2.638595, 2.682385, 2.69403, 2.873687, 3.022308, 3.230573,
                 3.638696, 3.771732, 3.991832, 4.059629, 4.203285, 4.475985, 4.615193, 5.033052,
                 5.04564, 5.073794, 5.179396, 6.160659, 6.601993, 6.617971, 8.257484, 8.427954,
                 8.783522, 10.829314])



# Compute the spacing between neighboring points
h = np.diff(data)
h_std = np.std(h)

target_start_client = int(len(data) * 3 / 5)
avg_diff = (10.829314 - data[target_start_client-1]) / (50 - target_start_client)
threshold = avg_diff
print(avg_diff)
print(h_std)
print(threshold)

# Compute the first derivative using the finite difference method
deriv = np.divide(h[:-1], h[1:] + h[:-1])
# deriv =

# Identify the points where the derivative is large or changes sign
sharp_points = np.where(h > threshold)[0]

# Print the indices of the sharp points
print(sharp_points)
