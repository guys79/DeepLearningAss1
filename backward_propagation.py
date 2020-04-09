import numpy as np

from forward_propagation import softmax


def Linear_backward(dZ, cache):
    """
    The implementation of the linear part of the backward propagation process for a single layer
    :param dZ: The derivative of the cos function with respect to Z
    :param cache: A tuple of (A_prev,W,b).
    :return: dA_prev, dW, db
    """
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    num_samples = A_prev.shape[1]

    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A_prev.T) / num_samples
    db = np.sum(dZ, axis=1, keepdims=True) / num_samples

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation="relu"):
    """
    The implementation for the LINEAR -> ACTIVATION layer. The function first computed dZ and then applies the
    linear_backward function
    :param dA: The derivative of the cos function with respect to A
    :param cache: A tuple of (A_prev,W,b)
    :param activation: either 'relu' or 'softmax'
    :return: dA_prev, dW, db
    """
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
    Implements backward propagation for a relu unit
    :param dA: post activation gradient of the current layer
    :param activation_cache: contains Z
    :return: derivative of relu
    """
    Z = activation_cache
    dZ = np.ones(Z.shape)
    dZ[Z <= 0] = 0
    return dZ * dA


def softmax_backward(dA, activation_cache):
    """
    Implements the backward propagation process for the entire network.
    :param dA: ignored
    :param activation_cache: contains Z and Y
    :return: derivative of softmax
    """
    Z, Y = activation_cache
    A = softmax(Z)['A']
    return A - Y


def L_model_backward(AL, Y, caches):
    """
    Implements the backward propagation process for the entire network
    :param AL: the probabilities vector, the output of the forward propagation (L_model_forward)
    :param Y: the true labels vector
    :param caches: list of caches containing for each layer
    :return: gradients for the update of weights
    """
    grads = {}  # The dictionary with the gradients
    num_layers = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = AL  # will be ignored for last layer
    
    # softmax layer
    layer_grads = linear_activation_backward(dAL, caches[num_layers - 1] + [Y], activation='softmax')
    grads['dA' + str(num_layers - 1)], grads['dW' + str(num_layers - 1)], grads['db' + str(num_layers - 1)] = layer_grads

    # relu layers
    for i in reversed(range(num_layers - 1)):
        cache = caches[i]
        layer_grads = linear_activation_backward(grads['dA' + str(i + 1)], cache)
        grads['dA' + str(i)], grads['dW' + str(i)], grads['db' + str(i)] = layer_grads
   
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates the parameters using gradient descent
    :param parameters: dictionary containing the DNN architecture’s parameters
    :param grads: dictionary containing the gradients
    :param learning_rate: the learning rate used to update the parameters
    """
    W = parameters['W']
    b = parameters['b']
    for i in range(len(b)):
        W[i] -= learning_rate * grads["dW" + str(i)]
        b[i] -= learning_rate * grads["db" + str(i)]
