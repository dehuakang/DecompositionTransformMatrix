# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:15:51 2017

@author: wjcheon
"""
#%%
import numpy as np
import xlrd
from itertools import takewhile
import math

ws = xlrd.open_workbook("vts_positions.xlsx")
ws = ws.sheet_by_index(0)
ncols = ws.ncols
nrows = ws.nrows
ws.col
print "-------- Sheet information --------"
print "Number of col: " + str(ncols)
print "Number of low: " + str(nrows)

#%
#cells = ws.row_slice(rowx=0, start_colx=0, end_colx= ncols)                              
#%                           
dataSet = [] 
for iter_rows in range(nrows):
    row_buffer = ws.row_slice(rowx=0, start_colx=0, end_colx= ncols)
    dataSet.extend(row_buffer)
    del row_buffer
#
dataSet = np.asarray(dataSet)
dataSet  = np.reshape(dataSet, [nrows, ncols])
vts_positions = np.matrix(dataSet)

#%%
ws = xlrd.open_workbook("transformed_positions.xlsx")
ws = ws.sheet_by_index(0)
ncols = ws.ncols
nrows = ws.nrows
ws.col
print "-------- Sheet information --------"
print "Number of col: " + str(ncols)
print "Number of low: " + str(nrows)

#%
#cells = ws.row_slice(rowx=0, start_colx=0, end_colx= ncols)                              
#%                           
dataSet = [] 
for iter_rows in range(nrows):
    row_buffer = ws.row_slice(rowx=0, start_colx=0, end_colx= ncols)
    dataSet.extend(row_buffer)
    del row_buffer
#
dataSet = np.asarray(dataSet)
dataSet  = np.reshape(dataSet, [nrows, ncols])
transformed_positions = np.matrix(dataSet)

#%%
import tensorflow as tf
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 4])

Rtheta_x = tf.Variable(tf.random_normal([1]), name='Rthata_x')
Rtheta_y = tf.Variable(tf.random_normal([1]), name='Rthata_y')
Rtheta_z = tf.Variable(tf.random_normal([1]), name='Rthata_z')
Trans_x = tf.Variable(tf.random_normal([1]), name='Trans_x')
Trans_y = tf.Variable(tf.random_normal([1]), name='Trans_y')
Trans_z = tf.Variable(tf.random_normal([1]), name='Trans_z')

R_mat_X = [[1, 0, 0, 0],[0, tf.cos(Rtheta_x)[0], -tf.sin(Rtheta_x)[0], 0], 
           [0, tf.sin(Rtheta_x)[0], tf.cos(Rtheta_x)[0], 0],[0, 0, 0, 1]]

R_mat_Y = [[tf.cos(Rtheta_y), 0, tf.sin(Rtheta_y), 0],[0, 1., 0, 0],
           [tf.matmul(-1.,tf.sin(Rtheta_y)), 0, tf.cos(Rtheta_y), 0],[0, 0, 0, 1]]

R_mat_Z = [[tf.cos(Rtheta_z), tf.matmul(-1., tf.sin(Rtheta_z)), 0, 0],[tf.sin(Rtheta_z), tf.cos(Rtheta_z), 0, 0],
           [0, 0, 1, 0],[0, 0, 0, 1]]


T_mat = [[1, 0, 0, Trans_x],[0, 1, 0, Trans_y],
           [0, 0, 1, Trans_z],[0, 0, 0, 1]]



# Hypothesis usiRng sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
Mat_set = tf.matmul(R_mat_Z, tf.matmul(R_mat_Y, tf.matmul(R_mat_X, X)))
hypothesis = (tf.matmul(Mat_set, X))

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

#%%











