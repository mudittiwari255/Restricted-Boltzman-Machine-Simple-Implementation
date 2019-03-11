#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 05:26:23 2019

@author: mudit
"""

import numpy as np
from numpy.matlib import repmat

lr = 0.1
weightcost = 0.0002
momentum = 0.001

def rbm(numcases : int, numdims : int, numhids : int, numbatches : int, n_epochs : int, batchdata : np.array):
    vishid = 0.1 * np.random.randn(numdims, numhids)
    hidbiases = np.zeros((1, numhids))
    visbiases = np.zeros((1, numdims))
    poshidprobs = np.zeros((numcases, numhids))
    neghidprobs = np.zeros((numcases,numhids))
    posprods    = np.zeros((numdims,numhids))
    negprods    = np.zeros((numdims,numhids))
    vishidinc  = np.zeros((numdims,numhids))
    hidbiasinc = np.zeros((1,numhids))
    visbiasinc = np.zeros((1,numdims))
    batchposhidprobs= np.zeros((numcases,numhids,numbatches))
    for i in range(n_epochs):
        print("Epoch {}".format(i))
        error_sum = 0
        for batch in range(numbatches):
            data = batchdata[:,:,batch]
            poshidprobs = 1 / (1 + np.exp(-data@vishid - repmat(hidbiases, numcases, 1)))
            batchposhidprobs[:,:,batch] = poshidprobs
            posprods = np.transpose(data) @ poshidprobs
            poshidact = sum(poshidprobs)#print(poshidact)
            posvisact = sum(data)
            poshidstates = poshidprobs > np.random.rand(numcases, numhids)
            negdata = 1 / (1 + np.exp(poshidstates @ np.transpose(-vishid) - repmat(visbiases, numcases, 1)))
            neghidprobs = 1 / (1 + np.exp(-negdata @ vishid - repmat(hidbiases, numcases, 1)))
            negprods = np.transpose(negdata) @ neghidprobs
            neghidact = sum(neghidprobs)
            negvisact = sum(negdata)#print(neghidact)
            error = sum(sum(np.square(data - negdata)))
            error_sum = error + error_sum
            vishidinc = momentum * vishidinc + lr * ((posprods - negprods) / numcases - weightcost * vishid)
            visbiasinc = momentum * visbiasinc + (lr / numcases) * (posvisact - negvisact)
            hidbiasinc = momentum * hidbiasinc + (lr / numcases) * (poshidact - neghidact)
            vishid = vishid + vishidinc
            visbiases = visbiases + visbiasinc
            hidbiases = hidbiases + hidbiasinc
        print("Epoch : {} and Error {}".format(i, error_sum))
    return hidbiases, vishid, visbiases, batchposhidprobs
