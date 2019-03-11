# Restricted-Boltzman-Machine-Simple-Implementation

Objective is to map images of MNIST dataset (each image comprises of 784 pixels) to a lower dimension by reducting the Reconstruction error. Also, we want to demonstrate the reconstruction of each image. All the codes written are based on this great [paper](https://www.cs.toronto.edu/~hinton/science.pdf).

# Usage
This Repo contains three files:
1) boltzman.py : Contains utility function for one RBM.
2) encoder.py : Contains utility functions for stacking RBMs. 
3) mnist.py : Contains the driver code to minimize reconstruction error and hence encoding the images.

Just keep every file in same path and run mnist.py. 

(Dependencies : Tensor-flow, numpy).

