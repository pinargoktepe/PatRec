import numpy as np
from shutil import copyfile
import os


def writeDataToFolder(listOfFiles, imgFolder, destFolder, ext):

    for file in listOfFiles:
        src = imgFolder + file + ext
        dest = destFolder + file + ext
        copyfile(src, dest)

def createFolder(folderName):

    if os.path.isdir(folderName) == False:
        try:
            os.mkdir(folderName)
        except OSError:
            print("Creation of the directory %s failed" % folderName)
        else:
            print("Successfully created the directory %s " % folderName)

    else:
        print("%s already exists!" % folderName)

def splitData(folder, train_file, val_file):

    train = open(train_file, "r")
    val = open(val_file, "r")

    lines_train = train.read().split('\n')
    lines_train = lines_train[0:len(lines_train)-1]
    lines_val = val.read().split('\n')
    lines_val = lines_val[0:len(lines_val)-1]
    train.close()
    val.close()

    #create training and validation folders
    dataset_root = os.path.join(folder, 'dataset/')
    gt = os.path.join(dataset_root, 'ground-truth/')
    train_folder = os.path.join(dataset_root, 'train/')
    validation_folder = os.path.join(dataset_root, 'validation/')
    train_folder_gt = os.path.join(gt, 'train/')
    validation_folder_gt = os.path.join(gt, 'validation/')
    createFolder(dataset_root)
    createFolder(train_folder)
    createFolder(validation_folder)
    createFolder(gt)
    createFolder(train_folder_gt)
    createFolder(validation_folder_gt)

    #Write images to realted folder under the dataset folder
    imageFolder = folder + "images/"
    gt_locationsFolder = folder + "ground-truth/locations/"
    writeDataToFolder(lines_train, imageFolder, train_folder, ".jpg")
    writeDataToFolder(lines_val, imageFolder, validation_folder, ".jpg")
    writeDataToFolder(lines_train, gt_locationsFolder, train_folder_gt, ".svg")
    writeDataToFolder(lines_val, gt_locationsFolder, validation_folder_gt, ".svg")

    return train_folder, lines_train, validation_folder, lines_val



#folder = "../../PatRec17_KWS_Data/"
#train_file = folder + "task/train.txt"
#val_file = folder + "task/valid.txt"

#splitData(folder, train_file, val_file)