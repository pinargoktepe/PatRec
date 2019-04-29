from preprocessing_main import preprocessing_main
from keyword_spotting import spot_keywords

folder = "../../PatRec17_KWS_Data/"
train_file = folder + "task/train.txt"
val_file = folder + "task/valid.txt"
binarization_method = "sauvola"
window_size = 15
preprocessing_main(folder, train_file, val_file, binarization_method, window_size)







