import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from operator import itemgetter 


def draw_boxes(img, windows, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
        # Draw a rectangle given bbox coordinates
   
    for window in windows:
        window = tuple(map(tuple, window))    # convertion from np array to tuple
        cv2.rectangle(imcopy, window[0], window[1], color, thick)
        # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None),
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    
    y_shape, x_shape, depth = img.shape
    # If x and/or y start/stop positions not defined, set to image size
    if (x_start_stop==(None, None)):
        x_start_stop = (0,x_shape)
    if (y_start_stop==(None, None)):
        y_start_stop = (0,y_shape)
    
    x_grid = np.arange(x_start_stop[0],x_start_stop[1]-xy_window[0],xy_window[0]*(1-xy_overlap[0])).astype(int)
    y_grid = np.arange(y_start_stop[0],y_start_stop[1]-xy_window[1],xy_window[1]*(1-xy_overlap[1])).astype(int)  
    
#    print(x_start_stop)  # Just for debugging
#    print(x_grid)
#    print(y_grid)
    # Initialize a list to append window positions to
#    window_list =[]
    window_list = []
    top_left =  (0,0)
    bottom_right = xy_window
    
    for y in y_grid[:-1]:
        for x in x_grid:
            top_left = (x,y)
            bottom_right = (x + xy_window[0],y + xy_window[1])
            window_list.append((top_left,bottom_right))  # Store window coordinates
    
    # Return the list of windows
    return window_list

##############################################################################
# Functions that search for car patterns over a image frame
# Optimized for GPU - Keras implementation
def search_cars(image,windows,model,show=False):

    img_search = np.copy(image)
    box_matches = []
    image_patches = []
    
    
    for window in windows:
        # Extract part of the image to the window
        x1 = window[0][0]
        x2 = window[1][0]
        y1 = window[0][1]
        y2 = window[1][1]
        test_img = img_search[y1:y2,x1:x2]
        
        image_patch = cv2.resize(test_img,(64,64))
        image_patches.append(image_patch)
        
        
    image_patches = np.array(image_patches)    
    test_prediction = model.predict_classes(image_patches,verbose = 0)
    
    matches_index = [i for i,v in enumerate(test_prediction) if v ==1]
    matches_index = np.array(matches_index)  # convertion to numpy matrix
    
    windows = np.array(windows)
    if (matches_index != []):
        box_matches = windows[matches_index]
    
    if (show==True):
        return draw_boxes(image,box_matches), box_matches
    else:
        return box_matches
                       
##############################################################################

def add_heat(image, bbox_list,threshold = 0):
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    
    heatmap[heatmap <= threshold] = 0
    # Return updated heatmap
    return heatmap

##############################################################################

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

##############################################################################
### TESTS
##############################################################################    
#
# # load json and create model
from keras.models import model_from_json
import cv2

json_file = open('car_detector4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("car_detector_weights4.h5")
print("Loaded model from disk")

image = cv2.cvtColor(cv2.imread('test/test1.jpg'), cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.title('Original image')
cropped_img = image[400:650,:]
plt.imshow(cropped_img)
plt.title('Cropped image')

windows = slide_window(image, x_start_stop=(None, None), y_start_stop=(400, 650), 
                    xy_window=(64,64), xy_overlap=(0.75, 0.75))

img_search, box_matches = search_cars(image,windows,model,show=True) 
plt.imshow(img_search) 
plt.title('Car detection matches')
#################################################################################
#         
#
#from scipy.ndimage.measurements import label
#
## Add heat to each box in box list
#heat = add_heat(image,box_matches,threshold = 4)
#
## Visualize the heatmap when displaying    
#heatmap = np.clip(heat, 0, 255)
#
## Find final boxes from heatmap using label function
#labels = label(heatmap)
#draw_img = draw_labeled_bboxes(np.copy(image), labels)
#
##fig = plt.figure()
##plt.subplot(121)
##plt.imshow(draw_img)
##plt.title('Car Positions')
##plt.subplot(122)
##plt.imshow(heatmap, cmap='hot')
##plt.title('Heat Map')
##fig.tight_layout()
#
#plt.imshow(draw_img)
#plt.title('Car Positions')
#
#plt.imshow(heatmap, cmap='hot')
#plt.title('Heat Map')