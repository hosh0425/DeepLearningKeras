# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 03:43:00 2019

@author: Hossein
"""

from keras import datasets , utils , Model , layers

sequential_model = False
"""
    data perperation
    
"""
(train_images , train_lables) , (test_images , test_lables) = datasets.cifar10.load_data()

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images /= 255
test_images /= 255

train_lables = utils.np_utils.to_categorical(train_lables)
test_lables = utils.np_utils.to_categorical(test_lables)

"""
    Model defintion
"""
if not sequential_model:
    netInput = layers.Input()
    conv1 = layers.Conv2D(32 , (3,3) , strides=2 , padding='same' , activation='sigmoid')
