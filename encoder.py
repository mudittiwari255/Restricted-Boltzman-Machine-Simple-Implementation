#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 13:05:47 2019

@author: mudit
"""

import tensorflow as tf
import numpy as np


def load_rbm_weight(weights_bias_dict, layer):
    """
    weight_bias_dict : {layer : { W : WEIGHT , b : BIAS}}
    Output : Self Explanatory
    }
    """
    return weights_bias_dict[layer]['W'], weights_bias_dict[layer]['b']

input_size = 784
x = tf.placeholder(tf.float32, [None, input_size])

def encoder(x, input_size, layer_sizes, weights_bias_dict):
    """
    Forward propogation.
    """
    
    next_layer_input = x
    encoding_matrices = []
    encoding_biases = []
    for i in range(len(layer_sizes)):
        weight, bias = load_rbm_weight(weights_bias_dict, i)
        encoding_matrices.append(weight)
        encoding_biases.append(bias)
    for i in range(len(layer_sizes)):
        W = encoding_matrices[i]
        b = encoding_biases[i]
        output = tf.nn.sigmoid(tf.add(tf.matmul(next_layer_input, W) , b))
        next_layer_input = output
    encoder_x = next_layer_input
    return encoder_x, encoding_matrices

def decoder(x, layer_sizes, encoding_matrices):
    """
    Forward propogation for decoder
    """
    layer_sizes.reverse()
    #print(layer_sizes)
    encoding_matrices.reverse()
    
    decoding_matrics = []
    decoding_biases = []
    
    next_layer_input = x
    
    for i, dim in enumerate(layer_sizes[1:] + [int(next_layer_input.get_shape()[1])]):
        #print(i,dim)
        W = tf.identity(tf.transpose(encoding_matrices[i]))
        #print(W.get_shape())
        b = tf.Variable(tf.zeros([int(W.get_shape()[1])], dtype=tf.float64))
        
        decoding_matrics.append(W)
        decoding_biases.append(b)
        
        output = tf.nn.sigmoid(tf.add(tf.matmul(next_layer_input, W) , b))
        next_layer_input = output
        
    encoding_matrices.reverse()
    decoding_matrics.reverse()
    
    reconstructed_x = next_layer_input
    
    return reconstructed_x, decoding_matrics, decoding_biases

def recostruct(encoded, weights, bias):
    """
    Reconstructor : Encoded -> Original
    Not Functional
    """
    weights.reverse()
    
    for i,item in enumerate(weights):
        encoded = encoded @ item.eval() + bias[i].eval()
    return encoded
