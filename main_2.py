# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:15:51 2017

@author: Wonjoong Cheon
"""
#%% Load data of VTS measured positions
import numpy as np      
import xlrd
from itertools import takewhile
import math
import os
from time import localtime, strftime
timeline = localtime()
filename_log = strftime("%y_%m_%d_%H_%M_%S", timeline) + '_cost_log.txt'
path_log = os.path.join(os.getcwd(),filename_log)

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

#%% Load data of transformed positions 
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
import numpy as np
import xlrd
import math


# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[4, None])
Y = tf.placeholder(tf.float32, shape=[4, None])

# Set Rotational angles and Translational magnitude as tf.Variable for training
# Rotaional angle
thetax =  tf.Variable(tf.random_normal([]), name= 'thetax')
thetay =  tf.Variable(tf.random_normal([]), name= 'thetay')
thetaz =  tf.Variable(tf.random_normal([]), name= 'thetaz')
# Translational magnitude
transx =  tf.Variable(tf.random_normal([]))
transy =  tf.Variable(tf.random_normal([]))
transz =  tf.Variable(tf.random_normal([]))

# Test using tf.constant
#thetax =  tf.constant(1., name= 'thetax')
#thetay =  tf.constant(2., name= 'thetay')
#thetaz =  tf.constant(3., name= 'thetaz')
#  #
#transx =  tf.constant(10.)
#transy =  tf.constant(20.)
#transz =  tf.constant(30.)

# Building a tensorflow graph  
rotation_matrix_x = tf.stack([tf.constant(1.0),tf.constant(0.0),tf.constant(0.0), tf.constant(0.0),
                               tf.constant(0.0),tf.cos(thetax), -tf.sin(thetax), tf.constant(0.0),
                               tf.constant(0.0),tf.sin(thetax), tf.cos(thetax), tf.constant(0.0),
                               tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)])

rotation_matrix_y = tf.stack([
                          tf.cos(thetay), tf.constant(0.0), tf.sin(thetay), tf.constant(0.0),
                          tf.constant(0.0),tf.constant(1.0),tf.constant(0.0), tf.constant(0.0),
                          -tf.sin(thetay), tf.constant(0.0), tf.cos(thetay), tf.constant(0.0),
                          tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)])

rotation_matrix_z = tf.stack([
                              tf.cos(thetaz), -tf.sin(thetaz),tf.constant(0.0), tf.constant(0.0),  
                              tf.sin(thetaz), tf.cos(thetaz),tf.constant(0.0), tf.constant(0.0),
                              tf.constant(0.0),tf.constant(0.0),tf.constant(1.0), tf.constant(0.0),
                              tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)])
    
translation_matrix = tf.stack([tf.constant(1.0), tf.constant(0.0), tf.constant(0.0), transx,
                                tf.constant(0.0), tf.constant(1.0), tf.constant(0.0), transy,
                                tf.constant(0.0), tf.constant(0.0), tf.constant(1.0), transz,
                                tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), tf.constant(1.0)])
    
rotation_matrix_x = tf.reshape(rotation_matrix_x, [4,4])
rotation_matrix_y = tf.reshape(rotation_matrix_y, [4,4])
rotation_matrix_z = tf.reshape(rotation_matrix_z, [4,4])
translation_matrix = tf.reshape(translation_matrix, [4,4])

hypothesis  = tf.matmul(rotation_matrix_z, tf.matmul(rotation_matrix_y, tf.matmul(rotation_matrix_x, tf.matmul(translation_matrix, X)  ) ))
 
hypothesis_onevec = tf.reshape(hypothesis, [-1]) 
Y_onevec = tf.reshape(Y, [-1])

# Define cost function as RMSE (root mean square error)
cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(Y_onevec, hypothesis_onevec))))
# Define loss function: RMSE + l2_loss
# l2_loss prevents exploding of weights
loss = cost + 0.01*(tf.nn.l2_loss(thetax+thetay+thetaz+transx+transy+transz))
# Select parameter optimzer as Adam
train = tf.train.AdamOptimizer(0.01).minimize(loss)
#%%
vts_positions_manual = [[0., 0., 0., 1.],
                 [1., 0., 0., 1.],
                 [0., 1., 0., 1.],
                 [1., 1., 0., 1.],
                 [0., 0., 1., 1.],
                 [1., 0., 1., 1.],
                 [0., 1., 1., 1.],
                 [1., 1., 1., 1.]]

vts_positions_manual = np.transpose(vts_positions_manual)
#
#
transformed_positions_manual = [[-23.583, 17.945, -22.841, 1.],
                 [-23.1718, 17.8871, -23.7511, 1.],
                 [-24.4175, 17.5189, -23.1920, 1.],
                 [-24.0055, 17.4601, -24.1013, 1.],
                 [-23.9514, 18.8482, -23.0666, 1.],
                 [-23.5394, 18.7894, -23.9759, 1.],
                 [-24.7852, 18.4212, -23.4168, 1.],
                 [-24.3732, 18.3625, -24.3261, 1.]]
transformed_positions_manual = np.transpose(transformed_positions_manual)

# log 
f_log = open(path_log, 'wb+')
#
# Launch graph
sess = tf.Session() 
# Initialize TensorFlow variables
sess.run(tf.global_variables_initializer())
print("theta X is {}".format(sess.run(thetax)))
for step in range(1000000):    
    cost_val, _, hypothesis_val = sess.run([loss, train, hypothesis], feed_dict={X: vts_positions_manual, Y: transformed_positions_manual})
    if step % 1 == 0:
        f_log.write('{0:4f}\n'.format(cost_val))
        f_log.flush()

#        print("theta X is {}\n".format(sess.run(thetax)))
#        print("theta Y is {}\n".format(sess.run(thetay)))
#        print("theta Z is {}\n".format(sess.run(thetaz)))
#        print("Trans X is {}\n".format(sess.run(transx)))
#        print("Trans Y is {}\n".format(sess.run(transy)))
#        print("Trans Z is {}\n".format(sess.run(transz)))
#        print("hypothesis is {}\n".format(hypothesis_val))
        print("cost is {}\n".format(cost_val))
#        print("vts_positions_manual is {}\n".format(vts_positions_manual))
#%%
f_log.close()