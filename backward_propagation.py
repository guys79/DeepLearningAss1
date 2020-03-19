import numpy as np
"""
The implementation of the linear part of the backward propagation process for a single layer
"""
def Linear_backward(dZ, cache):
    A_prev = cache[0]
    W = cache[1]
    b = cache[2]

    dA_prev = np.dot(W.T,dZ)
    dW = np.dot(A_prev.T,dZ)
    db = dZ

    return dA_prev,dW,db

"""
The implementation for the LINEAR -> ACTIVATION layer. The function first computed dZ and then applies the 
linear_backward function
"""
def linear_activation_backward(dA,cache,activation):

    activation_function = activation[0]
    Z = activation[1]
    dZ = 1
    dA_prev , dW, db = Linear_backward(dZ,cache)

    return dA_prev, dW, db

"""
This function implements backward propagation for a RELU unit
"""
def relu_backward(dA,activation_cache):
    return "not implemented"

"""
This function implements backward propagation for a RELU unit
"""
def softmax_backward(dA,activation_cache):
    return "not implemented"



