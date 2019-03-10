# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 01:57:51 2019

@author: Hossein
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils


"""
    Data preperation
    Loading
    normalizing
"""
(trainImages , trainLables) , (testImages , testLables) = mnist.load_data()

trainData = trainImages.reshape(60000,28,28,1)
testData = testImages.reshape(10000,28,28,1)

trainData = trainData.astype('float32')
testData = testData.astype('float32')

trainData /= 255
testData /= 255

trainLable = np_utils.to_categorical(trainLables)
testLable = np_utils.to_categorical(testLables)

"""
    creatig model
"""
from keras.layers import Conv2D , MaxPool2D , Input , Flatten , Dense
from keras.models import Model
import keras

#by using Pooling Layers
#myInput = Input(shape=(28,28,1))
#conv1= Conv2D(32 , (3,3) , padding='same' , activation='relu')(myInput)
#pool1= MaxPool2D(pool_size=(2,2))(conv1)
#conv2= Conv2D(64 , (3,3) , padding='same' , activation='relu')(pool1)
#pool2= MaxPool2D(pool_size=(2,2))(conv2)
#flat= Flatten()(pool2) #tabdile mikone be vector
#outlayre= Dense(10 , activation='softmax')(flat)

#without using pooling layers
myInput = Input(shape=(28,28,1))
conv1= Conv2D(32 , (3,3) , padding='same' , activation='relu' , strides=2)(myInput)
conv2= Conv2D(64 , (3,3) , padding='same' , activation='relu', strides=2)(conv1)
flat= Flatten()(conv2) #tabdile mikone be vector
outlayre= Dense(10 , activation='softmax')(flat)

myModel = Model(myInput , outlayre)
myModel.summary()
myModel.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
"""
    train model
"""
trainHistory = myModel.fit(trainData , trainLable , batch_size=128 , epochs=20)
history= trainHistory.history

import matplotlib.pyplot as plt
losses = history['loss']
accuracies = history['acc']

plt.plot(losses)
plt.figure()
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.plot(accuracies)

testLablePrediction = myModel.predict(testData)
import numpy as np
testLablePrediction=np.argmax(testLablePrediction , axis=1)

#i=0;
#for index in testLablePrediction:
#    if(index != testLables[i]):
#        plt.figure()
#        plt.xlabel("Reality: "+str(testLables[i]))
#        plt.ylabel("Prediction: "+ str(index))
#        plt.imshow(testImages[i])
#        print(i)
#    if(i>200):
#        break
#    i=i+1
#    







