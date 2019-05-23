from preprocessing_main import preprocessing_main
from keyword_spotting import spot_keywords, get_distances
from preprocessing import binarization, cropImage, scaleImage
from getData import splitData, createFolder
import os

folder = "../../TestKWS/"
test_file = folder + "task/test.txt"
folder_train = "../../PatRec17_KWS_Data"
train_file = folder_train + "/task/train.txt"
validation_file = folder_train+"/task/valid.txt"
keywords_file = folder + "task/keywords.txt"
binarization_method = "sauvola"
output_file = 'results.txt'

window_size = 15

test = open(test_file, "r")

files_test = test.read().split('\n')
files_test = files_test[0:len(files_test)-1]
#print(files_test)

train = open(train_file, "r")

files_train = train.read().split('\n')
files_train = files_train[0:len(files_train)-1]

val = open(validation_file, "r")

files_val = val.read().split('\n')
files_val = files_val[0:len(files_val)-1]
#print(files_train)

'''
print("Binarization of test set")
binarization(folder+'images/', files_test, binarization_method, window_size)
# Use polygons as clipping mask
for f in files_test:
    imgFolder = folder+'images/' + "/binarized/"
    imgFile = imgFolder + f + ".png"
    maskFile = folder + "ground-truth/locations/" + f + ".svg"
    croppedImg_folder = os.path.join(imgFolder, f)
    createFolder(croppedImg_folder)
    cropImage(maskFile, imgFile, croppedImg_folder)

#Normalize images to same width aka same sequence length
folder_list = [folder + "images/binarized/"]
scaleImage(folder_list)


print("Binarization of training set")
binarization(folder_train+"/dataset/train/", files_train, binarization_method, window_size)
# Use polygons as clipping mask
for f in files_train:
    imgFolder = folder_train+'/dataset/train/binarized/'
    imgFile = imgFolder + f + ".png"
    maskFile = folder_train + "/dataset/ground-truth/train/" + f + ".svg"
    croppedImg_folder = os.path.join(imgFolder, f)
    createFolder(croppedImg_folder)
    cropImage(maskFile, imgFile, croppedImg_folder)

#Normalize images to same width aka same sequence length
folder_list = [folder_train+"/dataset/train/binarized/", folder + "images/binarized/"]
scaleImage(folder_list)
'''

'''
print("Binarization of val set")
binarization(folder_train+"/dataset/validation/", files_val, binarization_method, window_size)
# Use polygons as clipping mask
for f in files_val:
    imgFolder = folder_train+'/dataset/validation/binarized/'
    imgFile = imgFolder + f + ".png"
    maskFile = folder_train + "/dataset/ground-truth/validation/" + f + ".svg"
    croppedImg_folder = os.path.join(imgFolder, f)
    createFolder(croppedImg_folder)
    cropImage(maskFile, imgFile, croppedImg_folder)

folder_list = [folder_train+"/dataset/train/binarized/",folder_train+"/dataset/validation/binarized/", folder + "images/binarized/"]
scaleImage(folder_list)

'''

keywords_f = open(keywords_file, "r")

keywords = keywords_f.read().split('\n')

keywords = keywords[0:len(keywords)-2]
print(keywords)

keyword_ids = []
for ind in range(len(keywords)):
    tab_ind = keywords[ind].find(' ')
    k = keywords[ind][tab_ind+1:]
    keyword_ids.append(keywords[ind][:tab_ind])
    keywords[ind] = k

print("keywords: ", keywords)
print("ids: ", keyword_ids)

test_set = folder + "images/binarized/scaled"
transcription_source_path = keywords_file
file = open(output_file,"w+")
counter = 10
for ind in range(len(keyword_ids)):
    print(str(counter)+' keyowrds left!')
    k = keyword_ids[ind]
    print(k)
    input_keyword = folder_train + "/dataset/train/binarized/scaled/" + k[:3] + '/' + k +".png" # "October"
    print("input keyword", input_keyword)
    #sorted_distances = get_distances(input_keyword, normalize=True, comparison_words_folder=test_set)
    # test set. Given as a path to the source folder including all to-be tested images
    spotted_keywords, result = spot_keywords(input_keyword, test_set)
    file.write(result)
    counter -= 1
# prints a list of (the top 10) sorted distances and the validation results (id, transcription, distance)

file.close()

