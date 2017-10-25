# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:19:34 2017

@author: Toshiharu
"""
MAKEGIF = 0

#from findLines import find_window_centroids, window_mask, fitLines, Line
from findLines3 import lineFullSearch, lineSearchGuided, Line

from imageProcessing import perspectiveCal,Calibration,birdEye, laneFindInit

from drawingMethods import drawRegion

from imageFilter import Multifilter

import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

calibration = 'camera_cal/calibration.p' 
perspective = 'camera_cal/perspective.p'

mtx,dist,M,Minv = laneFindInit(calibration,perspective)

def pipeLine(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    top_down = birdEye(dst,M)
    binary_warped = Multifilter(top_down,s_thresh=(190, 255),b_thresh=(140,200),l_thresh=(200,255),sxy_thresh=(30,100), draw=False)
    #fitLines(leftCurve,rightCurve,binary_warped, window_width=50, window_height=120, margin=50,max_offset=60, max_Roffset=500, draw = False)
    if(leftLine.detected==True):
        lineSearchGuided(binary_warped,leftLine,rightLine,margin =80,minPoints=100, maxOffset=650,minOffset=150,Rmax=30000,debug=0)
    else:
        lineFullSearch(binary_warped,leftLine,rightLine,nwindows = 9, windowWidth=100, minpix=50,maxOffset=650,minOffset=150,Rmax=30000,debug =0)    
    overImage = drawRegion(img,leftLine,rightLine,Minv,mtx,dist)
    
    return overImage

inputFolder = 'test_images'
outputFolder = 'output_images'

from moviepy.editor import VideoFileClip
from IPython.display import HTML


## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
white_output = 'test_video_output_lane.mp4'
clip1 = VideoFileClip('test_video.mp4')
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#    clip1 = VideoFileClip(videoInput)
leftLine = Line()
rightLine = Line()

#leftLine.x_factor = 3.7/480 # m/pixel
#rightLine.x_factor = 3.7/480
#
#leftLine.y_factor= 30/720 # m/pixel
#rightLine.y_factor = 30/720 # m/pixel

white_clip = clip1.fl_image(pipeLine) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

###############################################################################
### Make a GIF for documentation!
###############################################################################
if (MAKEGIF==True):
    from moviepy.editor import *
    
    clip = (VideoFileClip("test_video_output_lane.mp4")
            .subclip(6)
            .resize(0.3))
    clip.write_gif("test_video_output_lane.gif")
