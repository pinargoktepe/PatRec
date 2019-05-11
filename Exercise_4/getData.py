import os
import numpy as np

def getUsers(usersFile):

    users = open(usersFile, "r")
    users = users.read().split('\n')
    users = users[0:len(users)-1]
    print(users)
    return users


def getGT(gtFile):

    gt = open(gtFile, "r")
    gt_lines = gt.read().split('\n')
    gt_users, gt_sigs, gt_labels  = [], [], []
    print(gt_lines)
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

def main(usersFile, gtFile, enrollmentFolder, verificationFolder):

    #Get users
    users = getUsers(usersFile)

    #Get GT
    gt_users, gt_sigs, gt_labels = getGT(gtFile)
    print(gt_users[0:3])
    print(gt_sigs[0:3])
    print(gt_labels[0:3])

    #All enrollment files in enrollment folder
    enrollmentFiles = getAllFileNames(enrollmentFolder)
    print(len(enrollmentFiles))
    #Get content of an enrollment file
    data = np.loadtxt(enrollmentFiles[0])





sigVerFolder = "../../../Desktop/SignatureVerification/"
usersFile = sigVerFolder + "users.txt"
gtFile = sigVerFolder + "gt.txt"
enrollmentFolder = sigVerFolder + "enrollment/"
verificationFolder = sigVerFolder + "verification/"
main(usersFile, gtFile, enrollmentFolder, verificationFolder)
