#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 20:33:05 2019

@author: mudit
"""

import tensorflow as tf
import numpy as np
from boltzman import rbm
import os
from encoder import encoder, decoder


#Change Path Accordingly
path = '/home/mudit/lordofdata/sumansirproblem/dat'
os.chdir(path)

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets("MNIST", one_hot=True)

# Train Test splits
X_train, y_train, X_test, y_test = mnist_data.train.images, mnist_data.train.labels, mnist_data.test.images, mnist_data.test.labels

batch_size = 30

#batched = X_train[0:30]
#start = 30
#for i in range(X_train.shape[0] // batch_size):
#    end = start + batch_size
#    batched = np.dstack((batched, X_train[start : end]))

print("Creating Batches")
    
batched_x = mnist_data.train.next_batch(batch_size)[0]
batched_y = mnist_data.train.next_batch(batch_size)[1]

iters =  X_train.shape[0] // batch_size

for i in range(iters):
    batched_x = np.dstack((batched_x, mnist_data.train.next_batch(batch_size)[0]))
    batched_y = np.dstack((batched_y, mnist_data.train.next_batch(batch_size)[1]))

print("Batches Created")

if not os.path.exists("weights"):
    os.makedirs("weights")


print("Training RBMs")

numcases, numdims, numbatches = batched_x.shape

numhids = 1000

hidbiases, vishid, visbiases, batchposhidprobs = rbm(numcases, numdims, numhids, numbatches, 1, batched_x)

rbm1 = {'hidbiases' : hidbiases, 'weight' : vishid, 'visbiases' : visbiases}

np.save("./weights/rbm1.npy", rbm1)

numhids2 = 500

hidbiases, vishid, visbiases, batchposhidprobs = rbm(numcases, numhids, numhids2, numbatches, 1, batchposhidprobs)

rbm2 = {'hidbiases' : hidbiases, 'weight' : vishid, 'visbiases' : visbiases}

np.save("./weights/rbm2.npy", rbm2)

numhids3 = 250

hidbiases, vishid, visbiases, batchposhidprobs = rbm(numcases, numhids2, numhids3, numbatches, 1, batchposhidprobs)

rbm3 = {'hidbiases' : hidbiases, 'weight' : vishid, 'visbiases' : visbiases}

np.save("./weights/rbm3.npy", rbm3)

numhids4 = 30

hidbiases, vishid, visbiases, batchposhidprobs = rbm(numcases, numhids3, numhids4, numbatches, 1, batchposhidprobs)

rbm4 = {'hidbiases' : hidbiases, 'weight' : vishid, 'visbiases' : visbiases}

np.save("./weights/rbm4.npy", rbm4)


###Use only hidden biases

rbm1_weights_and_biases = np.load("./weights/rbm1.npy")
rbm2_weights_and_biases = np.load("./weights/rbm2.npy")
rbm3_weights_and_biases = np.load("./weights/rbm3.npy")
rbm4_weights_and_biases = np.load("./weights/rbm4.npy")

weight_bias_dict = {0 : {'W' : rbm1_weights_and_biases.tolist()["weight"],
                         'b' : rbm1_weights_and_biases.tolist()['hidbiases']},
                    1 : {'W' : rbm2_weights_and_biases.tolist()["weight"],
                         'b' : rbm2_weights_and_biases.tolist()['hidbiases']},
                    2 : {'W' : rbm3_weights_and_biases.tolist()["weight"],
                         'b' : rbm3_weights_and_biases.tolist()['hidbiases']},
                    3 : {'W' : rbm4_weights_and_biases.tolist()["weight"],
                         'b' : rbm4_weights_and_biases.tolist()['hidbiases']},
    }
                    
print("RBMs Trained")


##Defining Placeholder
X = tf.placeholder(tf.float64, [None, X_train.shape[1]])

encode_x, encode_matrix = encoder(X, X_train.shape[1], range(4), weight_bias_dict)
decode_x, dec_mat, dec_biases = decoder(encode_x, layer_sizes= [0,1,2,3], encoding_matrices= encode_matrix)

loss = tf.reduce_mean(tf.pow(X - decode_x, 2))
obtimizer = tf.train.RMSPropOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

print("Starting Training of auto encoder")

num_steps = 5000
for i in range(1, num_steps + 1):
    batch_x, _ = mnist_data.train.next_batch(30)
    
    _, l = sess.run([obtimizer, loss], feed_dict={X : batch_x})
    
    if i % 1000 == 0:
        print('Step %i : MiniBatch Loss : %f' % (i,l))
        
print("Training Completed")

"""
Uncomment the chunk of code bellow to see reconstruction in action.
Notice : It will create 20 png files in your directory, each file comprises two 10 different
tuples of original and reconstructed imanges.
"""



for i in range(10):
    
    batch_x, _ = mnist_data.train.next_batch(1)
    g = sess.run(decode_x, feed_dict={X: batch_x})
    
    
    from scipy.misc import toimage
    A_recon = toimage(g.reshape(28,28))
#    
    A_orig = toimage(batch_x.reshape(28,28))
    A_recon.save(str(i) + 'recon.png')
    A_orig.save(str(i) + 'orig.png')