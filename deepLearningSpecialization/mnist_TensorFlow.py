# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:19:31 2019

@author: Hossein
"""
import tensorflow as tf
import numpy as np
import tensorflow.contrib
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets("MNIST_data/" , one_hot = True) 
n_traint = mnist.train.num_examples
n_validation = mnist.validation.num_examples
n_test = mnist.test.num_examples

n_input = 28*28
n_hidden1 = 512
n_hidden2 = 256
n_hidden3 = 128
n_output = 10

learning_rate = 0.001
num_epochs = 5000
batch_size = 128
dropout = 0.5
costs = []

X = tf.placeholder(tf.float32 , shape=[None , n_input])
Y = tf.placeholder(tf.float32 , shape=[None , n_output])
keep_prop = tf.placeholder(tf.float32)


weights = {
    
   'W1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
   'W2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
   'W3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
   'W4': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}
biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'b4': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

Z1 = tf.add(tf.matmul(X , weights['W1']) , biases['b1'])
A1 = tf.nn.relu(Z1)
Z2 = tf.add(tf.matmul(A1 , weights['W2']) , biases['b2'])
A2 = tf.nn.relu(Z2)
layer_drop = tf.nn.dropout(A2, keep_prop)
Z3 = tf.add(tf.matmul(A2 , weights['W3']) , biases['b3'])
A3 = tf.nn.relu(Z3)
layer_drop = tf.nn.dropout(A3, keep_prop)
Z4 = tf.add(tf.matmul(A3 , weights['W4']) , biases['b4'])


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z4))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(Z4, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(num_epochs):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prop:dropout})

    # print loss and accuracy (per minibatch)
    if epoch%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prop:1.0})
        print("Iteration", str(epoch), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))
    if epoch%5==0:
        costs.append(minibatch_loss)
        
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prop:1.0})
print("\nAccuracy on test set:", test_accuracy)




