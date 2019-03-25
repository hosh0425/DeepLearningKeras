# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:30:08 2019

@author: Hossein
"""

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
#from tf_utils import load_dataset , random_mini_batches , convert_to_on_hot , predict

"""
Writing and running programs in TensorFlow has the following steps:

    1-Create Tensors (variables) that are not yet executed/evaluated.
    2-Write operations between those Tensors.
    3-Initialize your Tensors.
    4-Create a Session.
    5-Run the Session. This will run the operations you'd written above.
    Therefore, when we created a variable for the loss, we simply defined the loss as a function
    of other quantities, but did not evaluate its value.
    To evaluate it, we had to run init=tf.global_variables_initializer().
    That initialized the loss variable, and in the last line we were finally able to evaluate
    the value of loss and print its value.
"""
y_hat = tf.constant(36 , name='y_hat')
y = tf.constant(39 , name='y')

loss = tf.Variable((y-y_hat)**2 , name='loss')

"""
 A placeholder is an object whose value you can specify only later.
 To specify values for a placeholder,
 you can pass in values by using a "feed dictionary" (feed_dict variable).
"""
x = tf.placeholder(tf.int64 , name='x')


"""
 When init is run later (session.run(init)),
 the loss variable will be initialized and ready to be computed
"""
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(loss))
    print(session.run(2*x , feed_dict = {x:3}))
    session.close()    
    
    
    
    
def linear_function():
    
    np.random.seed(1)
    
    X = tf.constant(np.random.randn(3,1), name='X')
    W = tf.constant(np.random.randn(4,3), name='W')
    b = tf.constant(np.random.randn(4,1), name='b')
    Y = tf.Variable(tf.matmul(W,X) + b, name='Y')
    
    init = tf.global_variables_initializer()

    session = tf.Session()
    session.run(init)
    result = session.run(Y)
    return result
    
print( "result = " + str(linear_function()))
    

def sigmoid(z):
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    results -- the sigmoid of z
    """
    
    ### START CODE HERE ### ( approx. 4 lines of code)
    # Create a placeholder for x. Name it 'x'.
    x = tf.placeholder(tf.float32,name='x')

    # compute sigmoid(x)
    sigmoid = tf.sigmoid(x)

    # Create a session, and run it. Please use the method 2 explained above. 
    # You should use a feed_dict to pass z's value to x. 
    with tf.Session() as session:
        # Run session and call the output "result"
        result = session.run(sigmoid, feed_dict={x:z})
        session.close()
    
    ### END CODE HERE ###
    
    return result

print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))
    
    
    
    
    
    
    
