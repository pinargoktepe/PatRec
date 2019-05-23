from feature_extraction import extract_features, get_dtw_distance
import numpy as np 
import cv2
import glob, os



# path definitions (for testing)
folder_test = "../../TestKWS/"
folder_train = "../../PatRec17_KWS_Data/"
train_img_folder = folder_train + "/dataset/train/binarized/scaled"
#val_img_folder = img_folder + "/validation/binarized/scaled"
transcription_source_path = folder_test + "task/keywords.txt"

"""
Input: .txt file of all keywords in the shape of "....270-02-01 t-e-s-t...."
Returns: dictionary where the key is the "image name", i.e. 270-02-01 and the value the transcription
To find a matching keyword, travers the dict for a match and return the counterparty

"""
def get_transcriptions(txt_file):
    # open the txt file, read it, and store it in a simple list
    print("Reading transcriptions file...")
    f = open(txt_file, 'r')
    input_text = f.readlines()
    transcriptions = [text.strip() for text in input_text] # remove the unnecessary linebreaks
    f.close()
    #print(transcriptions)
    transcriptions = transcriptions[:len(transcriptions)-1]
    # as of now: a list where transcription[x] = "270-02-01 t-e-s-t"
    # split it further once again to obtain: transcription[x] = ["270-02-01", "t-e-s-t"]
    transcriptions = [row.split() for row in transcriptions]
    #print(transcriptions)
    # to end, convert it into a dictionary for easier matching (not a big list, so its ok)
    trans_dict = dict([])
    trans_dict = {trans[0]: trans[1] for trans in transcriptions}
    #print(trans_dict)
    return trans_dict


"""
Given an input image (word image path), this function will return the sorted distances closest to the input.

For each image pair, the extracted features (global features: # transitions, %black, gravity center) 
are calculated and then applied in a DTW algorithm to find the distance. 

REQUIRES PYTHON 3.5+ FOR RECURSIVE FILE SEARCH !!!
"""
def get_distances(keyword_image_path, normalize, comparison_words_folder):

    # extract the features from the given keyword to spot in the document
    keyword_features = extract_features(keyword_image_path, normalize)

    print("Calculating distances...")
    distances = []

    for compared_word in glob.glob(comparison_words_folder + "/**/*" + ".png"): # for each word in each subfolder (OS dependant?)
        print(compared_word)
        # 1) calculate the feature vector for the second image
        word_features = extract_features(compared_word, normalize)
        # 2) calculate the dtw distance
        dtw_distance = get_dtw_distance(keyword_features, word_features)
        distances.append([os.path.basename(compared_word), dtw_distance])


    sorted_distances = sorted(distances,key=lambda l:l[1])
    return sorted_distances 


"""
Main function of the post-processing part:
Obtains the transcriptions, extracts the features for each word comparison,
calculated the DTW distance and returns sorted distances.
Prints a top 10 rank list of the closest matches.

Input: keyword_image_path : The keyword to spot in the documents, given as a filepath
       validation_set: Compare the keyword to the word images in this folder(-path)

Output: a printed rank list. also returns the sorted distances for optional further tasks.
"""
def spot_keywords(keyword_image_path, test_set):

    # get the transcriptions
    transcriptions = get_transcriptions(transcription_source_path)

    # return sorted distances (and matching keywords) given an input keyword
    sorted_distances = get_distances(keyword_image_path, normalize = True, comparison_words_folder=test_set)

    print("sorted distances: ", sorted_distances)
    # return the top 10 (arbitrary, can be changed) results
    print("   ")
    print("Keyword spotting: Find the following keyword:")

    filename = os.path.basename(keyword_image_path)[:-4]
    keyword_transcription = transcriptions.get(filename)
    distance = 0
    print("ID: {0}, \t transcription: {1}, \t distance: {2:.2f}".format(filename, keyword_transcription, distance))

    print("\nTop 10 spotted keywords, closest first: \n")
    result = ''
    result += str(keyword_transcription)+", "
    for rank in range(len(sorted_distances)):
        filename = sorted_distances[rank][0] # the "filename", png extension removed below
        filename = filename[:-4]
        #keyword_transcription = transcriptions.get(filename) # get the transcription
        distance = sorted_distances[rank][1]
        result += filename+', '
        if rank < len(sorted_distances)-1:
            result += str(distance)+', '
        else:
            result += str(distance)+'\n'
        #print("ID: {0},\t transcription: {1:<30s} \t distance: {2:.2f}".format(filename, keyword_transcription, distance))
        #print("ID: {0}, \t transcription: {1}, \t distance: {2:.2f}".format(filename, keyword_transcription, distance))

    return sorted_distances, result #optional

    



