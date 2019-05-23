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

    #print("Calculating distances...")
    distances = []

    for ind in range(len(enrollment_files)):
        #get data from enrollment file
        en_data = np.loadtxt(enrollment_files[ind])
        # compute features for enrollment signatures of given user
        enrollment_features = extract_features(en_data, normalize)
        dtw_distance = get_dtw_distance(verification_features, enrollment_features)
        distances.append(dtw_distance)

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

def calculatePrecisionRecall(preds, gt, k):
    tp, fp, fn = 0, 0, 0
    precision, recall = [], []
    for p in range(k):
        if preds[p] == 'g':
            if preds[p] == gt:
                tp += 1
            else:
                fn += 1
        elif preds[p] == 'f':
            if preds[p] != gt:
                fp += 1

        if tp == 0:
            precision.append(0)
            recall.append(0)
        else:
            precision.append(float(tp/(tp+fp)))
            recall.append(float(tp/(tp+fn)))

    return precision, recall


def calculateMAP(preds, gt, k):

    precision, recall = calculatePrecisionRecall(preds, gt, k)
    map = float(sum([p * r for p, r in zip(precision, recall)])/k)
    return map


def makePrediction(distances, threshold):
    predictions = []
    #print("distances: ", distances)
    #label the signatures
    for k in range(len(distances)):
        if distances[k][3] <= threshold:
            predictions.append('g')
        else:
            predictions.append('f')

    return predictions


def getMAP(gt_users, gt_sigs, gt_labels, users, distances, threshold):
    map_values, predictions, gt_values = [], [], []
    #print(distances)

    predictions = makePrediction(distances, threshold)
    gt_user_ind = gt_users.index(distances[0][0])
    #print("gt_user ind: ", gt_user_ind)
    tmp_gt = gt_sigs[gt_user_ind:gt_user_ind+45]

    gt_sig_ind = tmp_gt.index(distances[0][1]) # use 0 or any other instance since the verification signature is the same for all instances in distances[ind]
    label = gt_labels[gt_user_ind + gt_sig_ind]
    map = calculateMAP(predictions, label, 5)
    map_values.append(map)
    return map_values

def main():
    output_file = "results_sig_ver.txt"
    sigVerFolder = "../../../Desktop/TestSignatures/"
    usersFile = sigVerFolder + "users.txt"
    #gtFile = sigVerFolder + "gt.txt"
    enrollmentFolder = sigVerFolder + "enrollment/"
    verificationFolder = sigVerFolder + "verification/"
    users, gt_users, gt_sigs, gt_labels, enrollmentFiles, verificationFiles = getData(usersFile, None, enrollmentFolder, verificationFolder) #gtFile=None
    users = users[0:2]
    result = ''
    distances, mAPs = [], []
    print("Calculating distances...")
    for ind in range(len(users)):
        result += str(users[ind])+", "
        print("User: ", users[ind])
        dst = []
        user_enrollments, enrollment_signature_ids = getFiles_byUser(users[ind], enrollmentFiles)
        user_verifications, verification_signature_ids = getFiles_byUser(users[ind], verificationFiles)
        #print("enrollments: ", user_enrollments)
        #print("verifications: ", user_verifications)
        #print("user: ", users[ind])
        #print("verifications ids: ", verification_signature_ids)
        for vs in range(len(verification_signature_ids)):
            distance = get_distances(users[ind], verification_signature_ids[vs], user_enrollments, user_verifications[vs])
            print("distances ", vs, ": ", distance)
            result += str(verification_signature_ids[vs]) + ", " + str(min(distance))
            if vs < len(verification_signature_ids)-1:
                result += ", "
            else:
                result += "\n"
                print(result)

    file = open(output_file,  "w+")
    file.write(result)
    file.close()

    '''
        for i in range(len(dst)):
            m = getMAP(gt_users, gt_sigs, gt_labels, users, dst[i], threshold=250)
            mAPs.append(m)
            #print("mAP: ", m)
        distances.append(dst)
    print("Calculating mAP..")
    final_map = float(np.sum(mAPs)/len(mAPs))
    print("final mAP: ", final_map)
    '''

main()








