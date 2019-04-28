from getData import writeDataToFolder, createFolder
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage import io
import os
import matplotlib.pyplot as plt
from svgpathtools import svg2paths
from svg.path import parse_path
import PIL.ImageOps as ImageOps

from PIL import Image, ImageDraw
import numpy as np

import glob
import re #regex interpretation



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


def cropImage(maskFile, imgFile, croppedImg_folder):

    img = Image.open(imgFile).convert("RGB")
    imArray = np.asarray(img)
    #Read svg file to get the path coordinates
    paths, attributes = svg2paths(maskFile)
    mask_polygon = []

    for k, v in enumerate(attributes):
        path = parse_path(v['d'])
        path.closed
        n = int(path.length())
        #calcualate the cartesian coordinates from complex ones. real part of the complex number is coordinate of x axis and
        #imaginary part corresponds to y-axis coordinates
        pts = [(p.real, p.imag) for p in (path.point(i / n) for i in range(0, n + 1))]
        mask_polygon.append(pts)

    for i in range(len(mask_polygon)):
        print(mask_polygon[i])
        #create mask image
        imgPolygon = Image.new('L', (imArray.shape[1], imArray.shape[0]))
        #create an image for storing the resulting image after mask is applied
        #newImArray = np.empty(imArray.shape, dtype='uint8')
        newImArray = np.copy(imArray)
        newImg_file = croppedImg_folder + "/" + str(i) + ".png"

        ImageDraw.Draw(imgPolygon).polygon(mask_polygon[i], outline=1, fill=1) # outline=1, fill=1
        new_polygon = np.array(imgPolygon)

        newImArray[:, :, 0] = np.multiply(newImArray[:, :, 0], new_polygon) #np.multiply(newImArray[:, :, 1], new_polygon * 255) # Apply mask
        newImArray[:, :, 1] = np.multiply(newImArray[:, :, 1], new_polygon) #np.multiply(newImArray[:, :, 2], new_polygon * 255)  # Apply mask
        newImArray[:, :, 2] = np.multiply(newImArray[:, :, 2], new_polygon) #np.multiply(newImArray[:, :, 3], new_polygon * 255)  # Apply mask

        #Image.fromarray(newImArray, "RGB").show()
        newIm = Image.fromarray(newImArray, "RGB")
        invert_im = ImageOps.invert(newIm)
        image_cropped = newIm.crop(newIm.getbbox())
        image_cropped.convert("L")
        image_cropped.save(newImg_file)