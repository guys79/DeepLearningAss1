import numpy as np

from forward_propagation import softmax

"""
The implementation of the linear part of the backward propagation process for a single layer

Input:
dZ - The derivative of the cos function with respect to Z
cache - A tuple of (A_prev,W,b). A_prev is the A values of the previous layer. W - The weight values of the current
layer, The b value of the current layer
"""


def Linear_backward(dZ, cache):
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    num_samples = A_prev.shape[1]

    dA_prev = np.dot(W.T, dZ)

    dW = 1. / num_samples * np.dot(dZ, A_prev.T)
    db = 1. / num_samples * np.sum(dZ, axis=1, keepdims=True)

    return dA_prev, dW, db


"""
The implementation for the LINEAR -> ACTIVATION layer. The function first computed dZ and then applies the 
linear_backward function

Input:
dA - The derivative of the cos function with respect to A
cache - A tuple of (A_prev,W,b). A_prev is the A values of the previous layer. W - The weight values of the current
layer, The b value of the current layer
Activation a tuple of ('activation function name' and the activation_cache), activation_cache - contains z

"""


def linear_activation_backward(dA, cache, activation="relu"):
    linear_cache, activation_cache = cache[0], cache[1]
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        Y = cache[2]
        dZ = softmax_backward(dA, [activation_cache, Y])
    else:
        raise ValueError('activation must be either "relu" or "softmax"')

    return Linear_backward(dZ, linear_cache)


def relu_backward(dA, activation_cache):
    """
    This function implements backward propagation for a relu unit

    Input:
    dA – post activation gradient of the current layer
    cache – contains both the linear cache and the activations cache

    """
    Z = activation_cache
    dZ = np.ones(Z.shape)
    dZ[Z <= 0] = 0
    return dZ * dA


def softmax_backward(dA, activation_cache):
    """
    Implement the backward propagation process for the entire network.
    Inputs:
    AL - the probabilities vector, the output of the forward propagation (L_model_forward)
    Y - the true labels vector (the "ground truth" - true classifications)
    Caches - list of caches containing for each layer: a) the linear cache; b) the activation cache
    """
    Z, Y = activation_cache
    A = softmax(Z)['A']
    return A - Y


def L_model_backward(AL, Y, caches):
    grads = {}  # The dictionary with the gradients
    num_layers = len(caches)
    Y = Y.reshape(AL.shape)

    # The dA for the last layer
    # dAL = - np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)
    # dAL = - np.divide(Y, AL) + np.divide(1 - Y, 1 - AL)
    dAL = AL

    # The last layer uses softmax
    grads["dA" + str(num_layers - 1)], grads["dW" + str(num_layers - 1)], grads["db" + str(num_layers - 1)] = \
        linear_activation_backward(dAL, caches[num_layers - 1] + [Y], activation="softmax")

    # Relu layers
    # From the last layer to the first
    for i in reversed(range(num_layers - 1)):
        cache = caches[i]
        grads["dA" + str(i)], grads["dW" + str(i)], grads[
            "db" + str(i)] = linear_activation_backward(grads["dA" + str(i + 1)], cache)
    return grads


"""
This function will update the parameters using gradient descent 

parameters – a python dictionary containing the DNN architecture’s parameters
grads – a python dictionary containing the gradients (generated by L_model_backward)
learning_rate – the learning rate used to update the parameters (the “alpha”)

"""


def update_parameters(parameters, grads, learning_rate):
    W = parameters['W']
    b = parameters['b']

    # Gradient step
    for i in range(len(b)):
        W[i] -= learning_rate * grads["dW" + str(i)]
        b[i] -= learning_rate * grads["db" + str(i)]
