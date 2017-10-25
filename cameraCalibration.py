# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:34:40 2017

@author: Toshiharu
"""

from imageProcessing import perspectiveCal,Calibration,birdEye
import matplotlib.pyplot as plt
import numpy as np
import cv2

Calibration(rows=6,cols=9,imagesFolder='camera_cal',show=False)

correction = 4
points_orig = np.float32([[(200, 720), (575+correction, 457), (715-correction,457), (1150, 720)]])
points_world = np.float32([[(440, 720), (440, 0), (950, 0), (950, 720)]])

perspectiveCal(points_orig,points_world,directory='camera_cal')

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
# Do camera calibration given object points and image points
chessBoard = cv2.cvtColor(cv2.imread('camera_cal/calibration1.jpg'), cv2.COLOR_BGR2RGB)
chessBoard_undist = cv2.undistort(chessBoard, mtx, dist, None, mtx)
cv2.imwrite('camera_cal/test_undist.jpg',chessBoard_undist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(chessBoard)
ax1.set_title('Chessboard Image', fontsize=15)
ax2.imshow(chessBoard_undist)
ax2.set_title('Undistorted Image', fontsize=15)
plt.show()

straightLines = cv2.cvtColor(cv2.imread('test_images/calib/straight_calib.jpg'), cv2.COLOR_BGR2RGB)
unwarped = birdEye(straightLines, mtx, dist,M)
cv2.imwrite('test_images/calib/straight_calib_unwarped.jpg',unwarped)

# Visualize unwarped
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(straightLines)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(unwarped)
ax2.set_title('Unwarped Image', fontsize=15)
plt.show()