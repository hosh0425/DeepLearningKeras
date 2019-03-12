# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 05:32:26 2019

@author: Hossein


DropOut: Simply put, dropout refers to ignoring units (i.e. neurons) during the training phase of certain set of neurons which is chosen at random.
         By “ignoring”, I mean these units are not considered during a particular forward or backward pass
         More technically, At each training stage, individual nodes are either dropped out of the net with probability 1-p or kept with probability p,
         so that a reduced network is left; incoming and outgoing edges to a dropped-out node are also removed.

         we use it to prevent over-fitting
         
         1- Dropout forces a neural network to learn more robust features
            that are useful in conjunction with many different random subsets of the other neurons.
        2- Dropout roughly doubles the number of iterations required to converge.
           However, training time for each epoch is less.
           
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1671)

NB_EPOCH = 100
BATCH_SIZE = 128 
VERBOSE = 1
NB_OUTPUT = 10
OPTIMIZER = SGD()
NB_HIDDEN = 128
DROPOUT=0.3
VALIDATION_SPLIT = 0.2

"""
load dataset
reshape
change type
normalize
transform categorical features into numerical vectors (One Hot Encoding)
"""
(X_train , Y_train) , (X_test , Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0] , X_test.shape[1]*X_test.shape[2])

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

"""
creating model & compile it
"""

model = Sequential()
model.add(Dense(NB_HIDDEN , input_shape=(X_train.shape[1],) , activation = 'relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_HIDDEN , activation='relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_OUTPUT , activation='softmax') )
model.summary()

"""
common choices for matrix
Accuracy: This is the proportion of correct predictions with respect to the targets
Precision: This denotes how many selected items are relevant for a multilabel classification
Recall: This denotes how many selected items are relevant for a multilabel classification
"""
model.compile(OPTIMIZER , loss=categorical_crossentropy , metrics=['accuracy'])


"""
train model

epochs: This is the number of times the model is exposed to the training set. At each iteration, the
        optimizer tries to adjust the weights so that the objective function is minimized.
        in other words, One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
        Since one epoch is too big to feed to the computer at once we divide it in several smaller batches

batch_size: This is the number of training instances observed before the optimizer performs a
            weight update. in other words: Since one epoch is too big to feed to the computer at once we divide it in several smaller batches
            you can’t pass the entire dataset into the neural net at once. So, you divide dataset into Number of Batches or sets or part

iteration: Iterations is the number of batches needed to complete one epoch.
            The number of batches is equal to number of iterations for one epoch    
            Let’s say we have 2000 training examples that we are going to use .
            We can divide the dataset of 2000 examples into batches of 500 then it will take 4 iterations to complete 1 epoch

We reserved part of the training set for validation. The key idea is that we reserve a
    part of the training data for measuring the performance on the validation while
    training. This is a good practice to follow for any machine learning task, which we
    will adopt in all our examples.
"""
history = model.fit(X_train , Y_train,
                        batch_size = BATCH_SIZE , epochs= NB_EPOCH,
                        verbose = VERBOSE , validation_split=VALIDATION_SPLIT)
    

"""
test the model

Note that the training set and the test set are, of course, rigorously separated. There is no point in
    evaluating a model on an example that has already been used for training. Learning is essentially a
    process intended to generalize unseen observations and not to memorize what is already known
"""
score = model.evaluate(X_test , Y_test , verbose=VERBOSE)
print("Test score: " , score[0])
print("Test accuracy: " , score[1])


"""
We have to add number of epochs till train accuracy intersect test accuracy
"""
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

































