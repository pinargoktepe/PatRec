import os
import numpy as np

def getUsers(usersFile):

    users = open(usersFile, "r")
    users = users.read().split('\n')
    users = users[0:len(users)-1]
    return users


def getGT(gtFile):

    gt = open(gtFile, "r")
    gt_lines = gt.read().split('\n')
    gt_lines = gt_lines[0:len(gt_lines) - 1]
    gt_users, gt_sigs, gt_labels  = [], [], []
    for e in gt_lines:
        ind_dash = e.find('-')
        ind_space = e.find(' ')
        gt_users.append(e[0:ind_dash])
        gt_sigs.append(e[ind_dash+1:ind_space])
        gt_labels.append(e[ind_space+1:])

    return gt_users, gt_sigs, gt_labels

def getAllFileNames(folderpath):

    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folderpath):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
    return files

def getData(usersFile, gtFile, enrollmentFolder, verificationFolder):

    #Get users
    users = getUsers(usersFile)

    #Get GT
    gt_users, gt_sigs, gt_labels = getGT(gtFile)

    #All enrollment files in enrollment folder
    enrollmentFiles = getAllFileNames(enrollmentFolder)
    verification_files = getAllFileNames(verificationFolder)

    return users, gt_users, gt_sigs, gt_labels, enrollmentFiles, verification_files

