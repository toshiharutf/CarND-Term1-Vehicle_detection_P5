# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 04:02:21 2017

@author: Toshiharu
"""

class Line():
    def __init__(self):
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations, only used in guided search
        self.best_fit = np.array([0,0,0])  
        #polynomial coefficients for the most recent fit
        self.current_fit = np.array([0,0,0]) 
        #radius of curvature of the line in some units
        self.radius = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.fitPoints = None  
        #Convertion form pixels to meters (x dir)
        self.x_factor = 3.7/480
        #Convertion form pixels to meters (y dir)
        self.y_factor = 30/720
        # bottom of the figure
        self.y_limit = 719
        # points to fit
        self.points = None
        self.yeval = 720/2
        
        self.detected = False  # Used to select between the full of guided seach
        self.timesMissed = 0  # Used only on the GuidedSearch function
    
#    def fitLine(self):
#        return  np.polyfit(self.fitPoints[:,1] , self.fitPoints[:,0], 2)
        
    def linePlot(self):
        ploty = np.linspace(0,self.y_limit-1, self.y_limit )
        plotx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        return plotx,ploty

    def basePos(self):
        return self.best_fit[0]*self.y_limit**2 + self.best_fit[1]*self.y_limit + self.best_fit[2]
    
    def basePosCurrent(self):
        return self.current_fit[0]*self.y_limit**2 + self.current_fit[1]*self.y_limit + self.current_fit[2]
    
   
#def findFull(warped_binary, nwindows = 9, windowWidth=100, minpix=50,draw = False):

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from imageFilter import Multifilter

#######################################################################################

def lineFullSearch(binary_warped,leftLine,rightLine,nwindows = 9, windowWidth=100, minpix=50, maxOffset=500,minOffset=200,Rmax =20000 ,debug = False): 
    smoothFactor = 0.9
    # Reset counter of missed lines
    leftLine.timesMissed = 0
    rightLine.timesMissed = 0
    
    # Generate an RGB image from a grayscaled one
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    histogram = np.sum(binary_warped[int((3/4)*binary_warped.shape[0]):,:], axis=0)
    midpoint = np.int(histogram.shape[0]/2)
    
    leftLine.base = np.argmax(histogram[:midpoint])
    rightLine.base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Reset initial position for each window
    leftx_current = leftLine.base
    rightx_current = rightLine.base
    
    leftCentroids=np.zeros((nwindows,2))   # Stores the centroids for each iterated window
    rightCentroids=np.zeros((nwindows,2))
    
    leftDetectedWindows = 0
    rightDetectedWindows = 0
    
    windowHeight = binary_warped.shape[0]/nwindows
    
    for window in range(nwindows):
    #window=0
        # Left Window
        leftBottom = int(binary_warped.shape[0]-window*windowHeight) 
        leftTop = int(binary_warped.shape[0]-(window+1)*windowHeight)
        leftLeft = int(leftx_current-windowWidth/2)
        leftRight = int(leftx_current+windowWidth/2)
        
        leftWindow = binary_warped[leftTop:leftBottom,leftLeft:leftRight]
        leftNPoints = np.sum(leftWindow)

        # Right Window
        rightBottom = int(binary_warped.shape[0]-window*windowHeight )
        rightTop = int(binary_warped.shape[0]-(window+1)*windowHeight)
        rightLeft = int(rightx_current-windowWidth/2)
        rightRight = int(rightx_current+windowWidth/2)
        
        rightWindow = binary_warped[rightTop:rightBottom,rightLeft:rightRight]
        rightNPoints = np.sum(rightWindow)
        
        if(leftNPoints>minpix):
            leftx_current = np.int(np.mean(np.nonzero(leftWindow)[1])) + leftLeft
            leftDetectedWindows += 1    # Update number of segments in left line
        
        if(rightNPoints>minpix):
            rightx_current = np.int(np.mean(np.nonzero(rightWindow)[1])) + rightLeft
            rightDetectedWindows += 1    # Update number of segments in left line
            
        # Store centroids of each window
        currentHeight = (leftTop+leftBottom)/2
        
        leftCentroids[window][0] = leftx_current 
        leftCentroids[window][1] = currentHeight
        
        rightCentroids[window][0] = rightx_current 
        rightCentroids[window][1] = currentHeight

   #    print(leftBottom, leftTop, leftLeft, leftRight)
    ############################################################################
    ### DEBUG
    ############################################################################
        
        if (debug== True):
            cv2.rectangle(out_img,(leftLeft,leftTop),(leftRight,leftBottom-1),
                (0,255,0), 4) 
            
            cv2.rectangle(out_img,(rightLeft,rightTop),(rightRight,rightBottom-1),
                (0,255,0), 4)
            
            print('Points in left window: ',window,leftNPoints)
            print('Points in right window: ',window,rightNPoints)


    ############################################################################
    ### DEBUG END
    ############################################################################ 
     
        # Fit a second order polynomial to each
    leftLine.current_fit  = np.polyfit(leftCentroids[:,1] , leftCentroids[:,0], 2)
    rightLine.current_fit = np.polyfit(rightCentroids[:,1], rightCentroids[:,0], 2)
    
    # Line smoothing
#    temp = (leftLine.current_fit[:2]+rightLine.current_fit[:2])/2
#    leftLine.current_fit[:2] = rightLine.current_fit[:2] = temp
    
                #Find the Line parameters in meters
    leftLineFit_world   = np.polyfit(leftCentroids[:,1]*leftLine.y_factor , leftCentroids[:,0]*leftLine.x_factor, 2)
    rightLineFit_world  = np.polyfit(rightCentroids[:,1]*rightLine.y_factor, rightCentroids[:,0]*rightLine.x_factor, 2)  
    y_eval = binary_warped.shape[0]/2*leftLine.y_factor  # evaluate at the middle of the figure
    leftLine.radius = ((1 + (2*leftLineFit_world[0]*y_eval + leftLineFit_world[1])**2)**1.5) / np.absolute(2*leftLineFit_world[0])
    rightLine.radius =  ((1 + (2*rightLineFit_world[0]*y_eval + rightLineFit_world[1])**2)**1.5) / np.absolute(2*rightLineFit_world[0])


    if( (   (rightLine.basePosCurrent()-leftLine.basePosCurrent())<maxOffset) & 
            ( (rightLine.basePosCurrent()-leftLine.basePosCurrent())>minOffset) &
            ( (rightLine.current_fit[2]-leftLine.current_fit[2] )<maxOffset) &
            ( (rightLine.current_fit[2]-leftLine.current_fit[2] )>minOffset) &
            (leftLine.radius < Rmax) &
            (rightLine.radius < Rmax) ) :
        
        leftLine.best_fit = leftLine.current_fit
        rightLine.best_fit = rightLine.current_fit
        

        
    if(leftDetectedWindows>4 ):
        leftLine.detected = True  # And go to the lineSearchGuided function 
#        leftLine.best_fit = leftLine.current_fit

    if( rightDetectedWindows>4 ):
        rightLine.detected = True
#        rightLine.best_fit = rightLine.current_fit
        
        
    ##########################################        
    if (debug== True):
        
        left_fitx,ploty = leftLine.linePlot()
        right_fitx,ploty = rightLine.linePlot()
        
        plt.plot(left_fitx, ploty, color='red')
        plt.plot(right_fitx, ploty, color='red')
        
        plt.imshow(out_img)
        plt.show()  

        print(leftLine.detected, rightLine.detected)    

###############################################################################
#### fullSearchLine() END
###############################################################################
 
    
    
def lineSearchGuided(binary_warped,leftLine,rightLine,margin = 100,minPoints=2000, maxOffset=500,minOffset=200,Rmax =20000 ,debug=False):
    smoothFactor = 0.7
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (leftLine.best_fit[0]*(nonzeroy**2) + leftLine.best_fit[1]*nonzeroy + 
    leftLine.best_fit[2] - margin)) & (nonzerox < (leftLine.best_fit[0]*(nonzeroy**2) + 
    leftLine.best_fit[1]*nonzeroy + leftLine.best_fit[2] + margin))) 
    
    right_lane_inds = ((nonzerox > (rightLine.best_fit[0]*(nonzeroy**2) + rightLine.best_fit[1]*nonzeroy + 
    rightLine.best_fit[2] - margin)) & (nonzerox < (rightLine.best_fit[0]*(nonzeroy**2) + 
    rightLine.best_fit[1]*nonzeroy + rightLine.best_fit[2] + margin))) 
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
 
        # Fit a second order polynomial to each
    if( (len(lefty)>0) & (len(righty)>0) ):
        leftLine.current_fit = np.polyfit(lefty, leftx, 2)
        rightLine.current_fit = np.polyfit(righty, rightx, 2)
        
         #Find the Line parameters in meters
        leftLineFit_world   = np.polyfit(lefty*leftLine.y_factor , leftx*leftLine.x_factor, 2)
        rightLineFit_world  = np.polyfit(righty*rightLine.y_factor,rightx*rightLine.x_factor, 2) 
    #    temp = (leftLineFit_world[:2]+rightLineFit_world[:2])/2
    #    leftLineFit_world[:2] = rightLineFit_world[:2] = temp
        y_eval = (binary_warped.shape[0]/2)*leftLine.y_factor  # evaluate at the middle of the figure
        leftLine.radius = ((1 + (2*leftLineFit_world[0]*y_eval + leftLineFit_world[1])**2)**1.5) / np.absolute(2*leftLineFit_world[0])
        rightLine.radius =  ((1 + (2*rightLineFit_world[0]*y_eval + rightLineFit_world[1])**2)**1.5) / np.absolute(2*rightLineFit_world[0])

    
    if( (   (rightLine.basePosCurrent()-leftLine.basePosCurrent())>maxOffset) | 
            ( (rightLine.basePosCurrent()-leftLine.basePosCurrent())<minOffset) |
            ( (rightLine.current_fit[2]-leftLine.current_fit[2] )>maxOffset) |
            ( (rightLine.current_fit[2]-leftLine.current_fit[2] )<minOffset) |
            (leftLine.radius > Rmax) |
            (rightLine.radius > Rmax) ) :
        
        leftLine.detected = False
        rightLine.detected = False  # Directly call for a fullSearchScan
        
        
    elif( (len(leftx)>minPoints) & (len(rightx)>minPoints) ) :  #
#        leftLine.best_fit = leftLine.current_fit
#        rightLine.best_fit = rightLine.current_fit
        
        # Interpolation with last known parameters
        leftLine.best_fit = leftLine.best_fit*(1-smoothFactor) + leftLine.current_fit*smoothFactor
        rightLine.best_fit = rightLine.best_fit*(1-smoothFactor) + rightLine.current_fit*smoothFactor

        
    else:
        leftLine.timesMissed +=1
        rightLine.timesMissed +=1
        
        if(leftLine.timesMissed>4):
            leftLine.detected = False
            rightLine.detected = False


    
    # Line smoothing
#    temp = (leftLine.current_fit[:2]+rightLine.current_fit[:2])/2
#    leftLine.current_fit[:2] = rightLine.current_fit[:2] = temp
    
    if(debug==True):
        
        # Generate x and y values for plotting
        left_fitx,ploty = leftLine.linePlot()
        right_fitx,ploty = rightLine.linePlot()
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        
        plt.show()
    




################################################################################
##### Testing
################################################################################
## Load camera lens distortion correction parameters
#from imageProcessing import perspectiveCal,Calibration,birdEye        
#        
#import pickle
#
#calibration_par = 'camera_cal/calibration.p'
#dist_pickle = pickle.load( open( calibration_par, "rb" ) )
#mtx = dist_pickle["mtx"]
#dist = dist_pickle["dist"]
#
## Load perpective matrices
#perspective_mat = 'camera_cal/perspective.p'
#dist_pickle = pickle.load( open( perspective_mat, "rb" ) )
#M = dist_pickle["M"]
#Minv = dist_pickle["Minv"]
#
########################################################################################
#### BATCH TESTING
########################################################################################
#import glob
#
#images = []
##images = glob.glob('test_images/straight_lines*.jpg')
#images = glob.glob('test_images/test*.jpg')
#
##calibration = {'mtx':mtx,'dist':dist,'M':M,'Minv':Minv}
##mtx = parameters['mtx']
##dist = parameters['dist']
##M = parameters['M']
##Minv = parameters['Minv']
#
#for idx, fname in enumerate(images):
#    print(fname)
#    img = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
#    top_down = birdEye(img, mtx, dist,M)
#    binary_warped = Multifilter(top_down,s_thresh=(210, 255),b_thresh=(155,200),l_thresh=(220,255),sxy_thresh=(30,100), draw=False)
#    
#    leftLine = Line()
#    rightLine = Line()
#    
#    lineFullSearch(binary_warped,leftLine,rightLine,nwindows = 9, windowWidth=100, minpix=50,debug = True)
#    lineSearchGuided(binary_warped,leftLine,rightLine,margin = 100, minPoints=2000,debug=True)
#    print(leftLine.radius, rightLine.radius)
