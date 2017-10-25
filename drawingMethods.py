# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 22:54:40 2017

@author: Toshiharu
inputs: img, curve.fitPoints, Minv
"""

import numpy as np
import cv2

# Create an img to draw the lines on
def drawRegion(img,leftCurve,rightCurve,Minv,mtx,dist):
    color_warp = np.zeros_like(img).astype(np.uint8)
        
    # Recast the x and y points into usable format for cv2.fillPoly()
    left_fitx, ploty  = leftCurve.linePlot()
    right_fitx, ploty = rightCurve.linePlot()
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank img
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original img space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original img
	
#    undist = cv2.undistort(img, mtx, dist, None, mtx)
    undist = img
    output = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    # Overlay text
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,50)
    fontScale              = 1
    fontColor              = (255,255,0)
    lineType               = 1
    text = '{}{:.2f}{}'.format('R curvature: ',rightCurve.radius, ' m') 
    
    cv2.putText(output,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,50+40)
    fontScale              = 1
    fontColor              = (255,255,0)
    lineType               = 1
    text = '{}{:.2f}{}'.format('L curvature: ',leftCurve.radius, ' m') 
    
    cv2.putText(output,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,50+40*2)
    fontScale              = 1
    fontColor              = (255,255,0)
    lineType               = 1
    x_offset = ((rightCurve.basePos()+leftCurve.basePos())/2 -img.shape[1]/2)*leftCurve.x_factor
    text = '{}{:.2f}{}'.format('Center offset: ',x_offset, ' m') 
    
    cv2.putText(output,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    
    
    return output
