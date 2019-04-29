from getData import splitData, createFolder
from preprocessing import binarization, cropImage, scaleImage
import os

def preprocessing_main(folder, train_file, val_file, binarization_method, window_size):
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
    #PG: There is problem related to loading folders from a list. That is why i separately run the scale image function for each folder.
    #    Otherwise it only scales images in the last folder
    folder_list = [train_folder + "binarized/"]
    scaled_train_files = scaleImage(folder_list)
    folder_list = [validation_folder + "binarized/"]
    scaled_validation_files = scaleImage(folder_list)

    #DTW and feature extraction
    train_img_folder =  "PatRec17_KWS_Data/dataset/train/scaled"
    validation_img_folder = "PatRec17_KWS_Data/dataset/validation/scaled"


