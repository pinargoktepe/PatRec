from getData import splitData, createFolder
from preprocessing import binarization, cropImage, scaleImage
import os



def main(folder, train_file, val_file, binarization_method, window_size):
    #If binarization method is Otsu, then window size will not be used

    #Split the i mage files as training and validation
    train_folder, files_train, validation_folder, files_val = splitData(folder, train_file, val_file)

    #Apply requested method for binarization of images in training and validation folders.
    print("Binarization of training set")
    binarization(train_folder, files_train, binarization_method, window_size)
    print("Binarization of validation set")
    binarization(validation_folder, files_val, binarization_method, window_size)

    #Use polygons as clipping mask
    for f in files_train:
        imgFolder = train_folder + "/binarized/"
        imgFile = imgFolder + f + ".png"
        maskFile = folder + "ground-truth/locations/" + f + ".svg"
        croppedImg_folder = os.path.join(imgFolder, f)
        createFolder(croppedImg_folder)
        cropImage(maskFile, imgFile, croppedImg_folder)
    
    for f in files_val:
        imgFolder = validation_folder + "/binarized/"
        imgFile = imgFolder + f + ".png"
        maskFile = folder + "ground-truth/locations/" + f + ".svg"
        croppedImg_folder = os.path.join(imgFolder, f)
        createFolder(croppedImg_folder)
        cropImage(maskFile, imgFile, croppedImg_folder)
    
    #Normalize images to same width aka same sequence length
    folder_list = [train_folder + "binarized/", validation_folder + "binarized/"]
    scaleImage(folder_list)



folder = "PatRec17_KWS_Data/"
train_file = folder + "task/train.txt"
val_file = folder + "task/valid.txt"
binarization_method = "sauvola"
window_size = 15


main(folder, train_file, val_file, binarization_method, window_size)
