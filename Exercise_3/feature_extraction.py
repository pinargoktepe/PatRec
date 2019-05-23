import numpy as np 
import cv2
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
def extract_bw_transitions(window):
    transitions = 0
    for x in range(0, len(window)-1):
        if window[x] != window[x+1]:
            transitions += 1
    return transitions



# percentage of black pixels within the sliding window
def extract_black_fraction(window):
    fraction = 0
    black_pixels = 0

    for pixel in range(0, len(window)):
        if window[pixel] == 0: #black
            black_pixels += 1

    # returns the percentage as a whole number, i.e. 20.3% not as 0.203, but "20" (no decimals)
    fraction = int(black_pixels / len(window) * 100)
    return fraction

def extract_gravity_center(window):
    fraction = 0
    black_pixels = 0
    total_sum = 0

    for pixel in range(0, len(window)):
        if window[pixel] == 0: # black
            black_pixels += 1
            total_sum += pixel

    if black_pixels == 0:
        gravity_center = 0
    else:
        gravity_center = int(total_sum / black_pixels)

    return gravity_center



# given an input of an image (path), extract and normalize (optional) the image features
def extract_features(image_path, normalize=True):
    #print("extract features: ",image_path)
    image = cv2.imread(image_path) # your image path

    #image = cv2.resize(image, dsize=(100, 100), interpolation=cv2.INTER_LINEAR)  ### quick simple alternative to scaling (if errors)
    image = np.asarray(image)
    #print("image shape: ", image.shape)
    image_height = image.shape[0]
    image_width = image.shape[1]
    stepSize = 1 # 1px offset
    (window_width, window_height) = (1, image_height) # window size


    transition_features = []
    fraction_features = []
    gravities = []

    for x in range(0, image_width-1, stepSize):
        # for each 1x100 (width x height) sliding window, offset by window_widht = 1px slider
        window = image[:window_height, x, 0]

        transition_features.append(extract_bw_transitions(window))
        fraction_features.append(extract_black_fraction(window))
        gravities.append(extract_gravity_center(window))
        

    image_features = np.array([transition_features, fraction_features, gravities])


    if normalize is True:
        image_features = image_features - np.mean(image_features)
        image_features /= np.std(image_features)        

    return image_features
