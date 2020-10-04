# opencv saturation correction - https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# modified by gs
# takes a file directory and runs gamma correction on all images
# in dir and saves to another dir

from __future__ import division
import cv2 as cv
import numpy as np
import os

# in dir
data_in_dir = "./"

dataLoader_dict = [x for x in os.listdir(data_in_dir) if x.endswith(".jpg")]

gamma = 2
gamma_max = 200

def gammaCorrection():
    # [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv.LUT(img_original, lookUpTable)
    # [changing-contrast-brightness-gamma-correction]
    img_gamma_corrected = res
    cv.imwrite('mod' + x, img_gamma_corrected)
    #cv.imshow("Gamma correction", img_gamma_corrected)

# def on_gamma_correction_trackbar(val):
#     global gamma
#     gamma = val / 100
#     gammaCorrection()

# process all images in the directory

for x in dataLoader_dict:

    # read in the image
    img_original = cv.imread(x)
    # error checking
    if img_original is None:
        print('Could not open or find the image: ', img_original)
        exit(0)
    # create array for new image
    img_gamma_corrected = np.empty(img_original.shape, img_original.dtype)

    # set the gamma value
    gamma_init = int(gamma * 100)
    # call gamma correction function
    #on_gamma_correction_trackbar(gamma_init)
    gammaCorrection()
    # save the corrected image

    #cv.createTrackbar('Gamma correction', 'Gamma correction', gamma_init, gamma_max, on_gamma_correction_trackbar)
    #cv.imwrite('mod' + x, img_gamma_corrected)

#cv.waitKey()
