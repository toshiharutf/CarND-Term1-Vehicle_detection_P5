# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 19:26:33 2017

@author: Toshiharu
"""
from imageProcessing import perspectiveCal,Calibration,birdEye
import cv2
import matplotlib.pyplot as plt

#Calibration(rows=6,cols=9,imagesFolder='camera_cal',show=False)
#
#correction = 4
#points_orig = np.float32([[(200, 720), (575+correction, 457), (715-correction,457), (1150, 720)]])
#points_world = np.float32([[(440, 720), (440, 0), (950, 0), (950, 720)]])
#
#perspectiveCal(points_orig,points_world,directory='camera_cal')


###########################################################################################
# Load camera lens distortion correction parameters
import pickle

calibration_par = 'camera_cal/calibration.p'
dist_pickle = pickle.load( open( calibration_par, "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Load perpective matrices
perspective_mat = 'camera_cal/perspective.p'
dist_pickle = pickle.load( open( perspective_mat, "rb" ) )
M = dist_pickle["M"]
Minv = dist_pickle["Minv"]

############################################################################################

import numpy as np
import cv2
import glob

images = []
#images = glob.glob('colorSpaces/*.png')
images = glob.glob('test_images/challenge/challenge*.jpg')
# Step through the list and search for chessboard corners
from imageFilter import Multifilter

for idx, fname in enumerate(images):
    print(fname)
    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
    top_down = birdEye(img, mtx, dist,M)
    filtered_img = Multifilter(top_down,s_thresh=(150, 255),b_thresh=(140,200),l_thresh=(225,255),sxy_thresh=(30,100), draw=True)
    filtered_rgb = np.dstack((filtered_img, filtered_img,filtered_img))*255
    plt.show()

