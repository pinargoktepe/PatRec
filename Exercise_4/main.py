from getData import getData
from feature_extraction import extract_features, get_dtw_distance
import numpy as np
import os

def get_distances(user_id, ver_sig_id, enrollment_files, verification_file, normalize=True):

    #get data in given verification file
    verification_data = np.loadtxt(verification_file)
    # compute features for given verification signature
    verification_features = extract_features(verification_data, normalize)
    #print("verification_features: ", verification_features)

    print("Calculating distances...")
    distances = []

    for ind in range(len(enrollment_files)):
        #get data from enrollment file
        en_data = np.loadtxt(enrollment_files[ind])
        # compute features for enrollment signatures of given user
        enrollment_features = extract_features(en_data, normalize)
        dtw_distance = get_dtw_distance(verification_features, enrollment_features)
        distances.append([user_id, ver_sig_id, ind, dtw_distance]) # [user number, verification signature number, enrollment signature number, distance]-> eg: [001, 01, 01, distance]

    sorted_distances = sorted(distances,key=lambda l:l[3])
    return sorted_distances


def getFiles_byUser(user_id, files):
    user_files = []
    signature_ids = []
    for f in files:
        ind_slash = f.rfind('/')
        ind_dash = f.find('-') #This is for getting the user name
        if f[ind_slash+1:ind_dash] == user_id:
            user_files.append(f)
            ind_dash = f.rfind('-') #this is for signature id
            ind_dot = f.rfind('.')
            signature_ids.append(f[ind_dash+1:ind_dot])

    return user_files, signature_ids


def main():

    sigVerFolder = "../../../Desktop/SignatureVerification/"
    usersFile = sigVerFolder + "users.txt"
    gtFile = sigVerFolder + "gt.txt"
    enrollmentFolder = sigVerFolder + "enrollment/"
    verificationFolder = sigVerFolder + "verification/"
    users, gt_users, gt_sigs, gt_labels, enrollmentFiles, verificationFiles = getData(usersFile, gtFile, enrollmentFolder, verificationFolder)
    users = users[0:2]
    for ind in range(len(users)):
        print("user: ", users[ind])
        user_enrollments, enrollment_signature_ids = getFiles_byUser(users[ind], enrollmentFiles)
        user_verifications, verification_signature_ids = getFiles_byUser(users[ind], verificationFiles)
        #print("enrollments: ", user_enrollments)
        #print("verifications: ", user_verifications)
        #print("verifications ids: ", verification_signature_ids)
        for vs in range(len(verification_signature_ids)):
            distances = get_distances(users[ind], verification_signature_ids[vs], user_enrollments, user_verifications[vs])
            print("distances: ", distances)


main()








