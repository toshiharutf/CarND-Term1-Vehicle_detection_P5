# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:36:58 2017

@author: Toshiharu
"""

MAKEGIF = 0

from image_match import *
import numpy as np

from findLines3 import lineFullSearch, lineSearchGuided, Line

from imageProcessing import perspectiveCal,Calibration,birdEye, laneFindInit

from drawingMethods import drawRegion

from imageFilter import Multifilter

import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# load camera calibration parameters
calibration = 'camera_cal/calibration.p' 
perspective = 'camera_cal/perspective.p'

mtx,dist,M,Minv = laneFindInit(calibration,perspective)

# load deep learning model and weights
from keras.models import model_from_json

# Load Deep learning NN model
json_file = open('car_detector4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("car_detector_weights4.h5")
print("Loaded model from disk")

from scipy.ndimage.measurements import label

#######################################################
##### PIPELINE
#######################################################

def pipeLine(img):
    
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    #######################################################
    ##### LANE DETECTION
    #######################################################
    top_down = birdEye(undist,M)
    binary_warped = Multifilter(top_down,s_thresh=(190, 255),b_thresh=(140,200),l_thresh=(200,255),sxy_thresh=(30,100), draw=False)
    #fitLines(leftCurve,rightCurve,binary_warped, window_width=50, window_height=120, margin=50,max_offset=60, max_Roffset=500, draw = False)
    if(leftLine.detected==True):
        lineSearchGuided(binary_warped,leftLine,rightLine,margin =80,minPoints=100, maxOffset=650,minOffset=150,Rmax=30000,debug=0)
    else:
        lineFullSearch(binary_warped,leftLine,rightLine,nwindows = 9, windowWidth=100, minpix=50,maxOffset=650,minOffset=150,Rmax=30000,debug =0)    
    
    #######################################################
    ##### CAR DETECTION
    #######################################################
    windows = slide_window(undist, x_start_stop=(None, None), y_start_stop=(400, 650), 
                xy_window=(64,64), xy_overlap=(0.75, 0.75))
    box_matches = search_cars(undist,windows,model,show=False)               
 
    # Add heat to each box in box list
    heat = add_heat(undist,box_matches,threshold = 3)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(undist), labels)
#    overImage = draw_img
#    draw_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    overImage = drawRegion(draw_img,leftLine,rightLine,Minv,mtx,dist)
    
    return overImage

##############################################################################
### VIDEO PROCESSING
##############################################################################    


from moviepy.editor import VideoFileClip
from IPython.display import HTML
import time

#    white_output = outputFolder + "/" + file
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
white_output = 'project_video_output.mp4'
clip1 = VideoFileClip('project_video.mp4')#.subclip(44,50)
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#    clip1 = VideoFileClip(videoInput)
leftLine = Line()
rightLine = Line()
#clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
#    clip1 = VideoFileClip(videoInput)

#leftLine.x_factor = 3.7/480 # m/pixel
#rightLine.x_factor = 3.7/480
#
#leftLine.y_factor= 30/720 # m/pixel
#rightLine.y_factor = 30/720 # m/pixel
t1 = time.time()

white_clip = clip1.fl_image(pipeLine) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

t2 = time.time()
print('Video processing time:', t2-t1)

###############################################################################
### Make a GIF for documentation!
###############################################################################
if (MAKEGIF==True):
    from moviepy.editor import *
    
    clip = (VideoFileClip("project_video_output.mp4")
            .subclip(27,40)
            .resize(0.3))
    clip.write_gif("project_video_output.gif")