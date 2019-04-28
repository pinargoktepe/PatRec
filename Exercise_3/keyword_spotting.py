from feature_extraction import extract_features, get_dtw_distance
import numpy as np 
import cv2
import glob, os


img_folder = "PatRec17_KWS_Data/dataset"
train_img_folder = img_folder + "/train/scaled"
val_img_folder = img_folder + "/validation/scaled" 


"""
Given an input image (word image path), this function will return the [...]

For each image pair, the extracted features (global features, # transitions, %black, gravity center) 
are calculated and then applied in a DTW algorithm to find the distance. 
"""
def spot_keywords(keyword_image_path, normalize, comparison_words_folder):

    # extract the features from the given keyword to spot in the document
    keyword_features = extract_features(keyword_image_path, normalize)


    print("Calculating distances...")
    distances = []
    for compared_word in glob.glob(comparison_words_folder + "/*" + ".png"):
        print(compared_word)
        # 1) calculate the feature vector for the second image
        word_features = extract_features(compared_word, normalize)
        # 2) calculate the dtw distance
        dtw_distance = get_dtw_distance(keyword_features, word_features)
        distances.append([os.path.basename(compared_word), dtw_distance])


    print(sorted(distances,key=lambda l:l[1]))
    return 0  # return the image? position? nearest neighbors?


if __name__ == "__main__":

    keyword_image_path = train_img_folder + "/271/20.png"  # the input image, i.e. the word to spot in the document
    comparison_words_folder = train_img_folder + "/271/"   # the words from the document in which the word is being looked for (folder path)

    spot_keywords(keyword_image_path, normalize = True, comparison_words_folder=comparison_words_folder)

