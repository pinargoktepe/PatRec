from getData import splitData
from preprocessing import binarization

def main(folder, train_file, val_file, binarization_method, window_size):
    #If binarization method is Otsu, then window size will not be used

    #Split the i mage files as training and validation
    train_folder, files_train, validation_folder, files_val = splitData(folder, train_file, val_file)

    #Apply requested method for binarization of images in training and validation folders.
    binarization(train_folder, files_train, binarization_method, window_size)
    binarization(validation_folder, files_val, binarization_method, window_size)


folder = "../../PatRec17_KWS_Data/"
train_file = folder + "task/train.txt"
val_file = folder + "task/valid.txt"
binarization_method = "sauvola"
window_size = 15
main(folder, train_file, val_file, binarization_method, window_size)
