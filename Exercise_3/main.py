from preprocessing_main import preprocessing_main
from keyword_spotting import spot_keywords

"""
Pattern Recognition Exercise 3: Keyword spotting
Group : github.com/pinargoktepe/PatRec

"""
#preprocessing
folder = "PatRec17_KWS_Data/"
train_file = folder + "task/train.txt"
val_file = folder + "task/valid.txt"
binarization_method = "sauvola"

window_size = 15
preprocessing_main(folder, train_file, val_file, binarization_method, window_size)


#keyword spotting task:
img_folder = "PatRec17_KWS_Data/dataset"
train_img_folder = img_folder + "/train/binarized/scaled"
val_img_folder = img_folder + "/validation/binarized/scaled" 

# transcription file: formatting might be OS-dependant, use the uploaded transcription-fix.txt if an error occurs.
transcription_source_path = "PatRec17_KWS_Data/ground-truth/transcription.txt"

# the input keyword. Input as an image path, i.e. "keyword_image_path = train_img_folder + "/270/270-05-09.png""
input_keyword = keyword_image_path = train_img_folder + "/270/270-01-06.png" # "October"
# validation set. Given as a path to the source folder including all to-be validated images
validation_set = val_img_folder 

# prints a list of (the top 10) sorted distances and the validation results (id, transcription, distance)
spotted_keywords = spot_keywords(input_keyword, validation_set)




