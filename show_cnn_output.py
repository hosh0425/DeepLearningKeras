# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:19:00 2019

@author: Hossein
"""
car1 = "car1.jpg"
car2 = "car2.jpg"
cat1 = "cat1.jpg"
cat2 = "cat2.jpg"

import cv2
from keras import Model , layers
from keras.models import Sequential
from keras import optimizers , losses , initializers
import numpy as np 
import matplotlib.pyplot as plt

def show_cnn_model_result(image_batch):
    image = np.squeeze(image_batch , axis=0)
   # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.normalize(image.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Convert to normalized floating point

    print (image.shape)
    
   # image = image.reshape(image.shape[:2])
    plt.imshow( image)
 
    
org_img = cv2.imread(cat2)
org_img = cv2.resize(org_img , (320,240))
#org_img = cv2.cvtColor(org_img , cv2.COLOR_RGB2GRAY)
#org_img = cv2.cvtColor(org_img , cv2.COLOR_GRAY2RGB)
img = cv2.normalize(org_img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Convert to normalized floating point

plt.imshow(img)

myModel= Sequential()
myModel.add(layers.Conv2D(3,(3,3),input_shape=img.shape,activation='relu', 
                          kernel_initializer=initializers.glorot_normal(),
                          bias_initializer=initializers.Constant()
                          ))
#myModel = Model(myInput , conv1)
#myModel.summary()
#myModel.compile(optimizer=optimizers.Adam(),loss=losses.categorical_crossentropy,metrics=['accuracy'])

#cnn_model.compile(optimizer=optimizers.Adam() , loss=losses.categorical_crossentropy , metrics=['accuracy'])


# keras expects batch of images, so we have to add a dimension
img_batch = np.expand_dims(img , axis=0)
conv_img = myModel.predict(img_batch)
show_cnn_model_result(conv_img)
