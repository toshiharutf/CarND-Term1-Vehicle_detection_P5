# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 16:36:57 2017

@author: Toshiharu
"""
EPOCHS = 10

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Conv2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping


from keras.regularizers import l2


model = Sequential()
model.add(Lambda(lambda x: (x/255.0-0.5)*2, input_shape=(64,64,3)))

# Layer 1
model.add(Conv2D(16,kernel_size=(3,3),activation='relu',subsample=(1,1),input_shape=(64,64,3)))  #(75-5+1)/2 = 36
model.add(MaxPooling2D(strides=(2,2)))

# Layer 2
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',subsample=(1,1)))  #(36-5+1)/2 = 16
model.add(MaxPooling2D(strides=(2,2)))

# Layer 3
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',subsample=(1,1)))  #(36-5+1)/2 = 16
model.add(MaxPooling2D(strides=(2,2)))

# Full dense layers
model.add(Flatten())
#model.add(Dense(1000, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))


#adam = Adam(lr=1e-03, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.7) 
#early_stopping = EarlyStopping(monitor='val_loss',
#                                   patience=1,
#                                   min_delta=0.00009)

model.compile('adam', 'categorical_crossentropy', ['accuracy'])


##########################################################################╚
########## LOAD DATA AND APPLY LABELS
###########################################################################
import numpy as np
import cv2
import sklearn
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

import matplotlib.image as mpimg

##############################################################################
import glob
cars = glob.glob('vehicles/**/*.png')
notcars = glob.glob('non-vehicles/**/*.png')
file_paths = cars + notcars

X_train = []

for path in file_paths:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    image = mpimg.imread(path)
    X_train.append(image)    

y_train = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

##############################################################################

encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = to_categorical(y_train)

X_train = np.array(X_train)

X_train, y_one_hot = sklearn.utils.shuffle(X_train, encoded_Y)

import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
plt.hist(y_one_hot,bins=50)
plt.show()

##########################################################################╚
########## TRAIN MODEL
###########################################################################

model.fit(X_train,y_one_hot,validation_split = 0.2, shuffle=True, epochs=EPOCHS)

##########################################################################╚
########## SAVE WEIGHTS AND MODEL
###########################################################################
# serialize model to JSON
model_json = model.to_json()
with open("car_detector4.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights('car_detector_weights4.h5')
print("Saved model to disk")

###########################################################################
########## SMALL TEST
###########################################################################
import glob

images = []
images = glob.glob('test/car*.png')

for image in images:
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(64,64))
    result = model.predict_classes(image[None, :, :, :], batch_size=1)
    print(result)
