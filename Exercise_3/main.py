from getData import splitData
from preprocessing import binarization

def main(folder, train_file, val_file):

    #Split the i mage files as training and validation
    train_folder, files_train, validation_folder, files_val = splitData(folder, train_file, val_file)

    #Apply Otsu's method for binarization of images in training and validation folders.
    binarization(train_folder, files_train)
    binarization(validation_folder, files_val)


folder = "../../PatRec17_KWS_Data/"
train_file = folder + "task/train.txt"
val_file = folder + "task/valid.txt"
main(folder, train_file, val_file)
