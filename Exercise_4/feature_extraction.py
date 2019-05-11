import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

"""
Implements the dynamic time warping alogrithm, based on the fastdtw library.
input: NUMPY(!) arrays of the extracted features from one word, and another word
returns the DTW path as well as the euclidian distance for later classification
"""


def get_dtw_distance(array1, array2):
    distance, path = fastdtw(array1, array2, dist=euclidean)
    return distance

########## feature extraction based on sliding windows: #########

# number of times of a color change within the window
def extract_velocity(prev_disp, disp, prev_time, time):

    velocity = float(abs(disp-prev_disp)/(time-prev_time))
    return velocity

# given input data, extract and normalize (optional) the data features
def extract_features(data, normalize=True):

    data = np.asarray(data)
    data_height = data.shape[0]
    data_width = data.shape[1]
    stepSize = 1  # 1px offset
    (window_width, window_height) = (1, data_height)  # window size

    x_features, y_features, vx_features, vy_features, pressure_features, time_features = [], [], [], [], [], []

    prev_x, prev_y, prev_time = 0.0, 0.0, 0.0
    for i in range(0, window_height - 1, stepSize):
        # for each 7x1 (width x height) sliding window, offset by window_height = 1px slider
        window = data[i, :data_width]
        #Separate the the window whose time feature is 0. (starting point)
        if window[0] == 0:
            prev_time = window[0]
            prev_x = window[1]
            prev_y = window[2]
        else:
            x_features.append(window[1])
            y_features.append(window[2])
            pressure_features.append(window[3])
            vx_features.append(extract_velocity(prev_x, window[1], prev_time, window[0]))
            prev_x = window[1]
            vy_features.append(extract_velocity(prev_y, window[2], prev_time, window[0]))
            prev_y = window[2]
            time_features.append(window[0]-prev_time)
            prev_time = window[0]

    data_features = np.array([x_features, y_features, vx_features, vy_features, pressure_features])
    data_features = data_features.T

    if normalize is True:
        mean_vals = np.mean(data_features, axis=0)
        std_vals = np.std(data_features, axis=0)
        for f in range(len(mean_vals)):
            data_features[:,f] = (data_features[:,f]-mean_vals[f])/std_vals[f]
    return data_features
