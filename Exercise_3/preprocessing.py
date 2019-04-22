from getData import writeDataToFolder, createFolder
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage import io
import os
import matplotlib.pyplot as plt
from PIL import Image

def binarization(folder, files, method, window_size):
    binarized_folder = os.path.join(folder, 'binarized/')
    createFolder(binarized_folder)

    for f in files:
        img_file = folder + f + ".jpg"
        img = io.imread(img_file)
        if method == "sauvola":
            thresh = threshold_sauvola(img, window_size)
        elif method == "otsu":
            thresh = threshold_otsu(img)

        binary_img = img > thresh
        bin_img_file = binarized_folder + f + ".png"
        binary_img = binary_img.astype('uint8')
        io.imsave(bin_img_file, 255 * binary_img)


def showImg(img, binary_img, thresh):
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')

    ax[1].hist(img.ravel(), bins=256)
    ax[1].set_title('Histogram')
    ax[1].axvline(thresh, color='r')

    ax[2].imshow(binary_img, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')
    plt.show()