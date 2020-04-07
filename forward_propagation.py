import numpy as np
from scipy.special import softmax as scipy_softmax

def initialize_parameters(layer_dims):
    """
    Initializes weights and biases for each layer
    :param layer_dims: an array of the dimensions of each layer in the network (layer 0 is the size of the flattened
    input, layer L is the output softmax)
    :return: a dictionary containing the initialized W and b parameters of each layer ( W1…WL, b1…bL)
    """
    W = []
    b = []
    for i in range(1, len(layer_dims)):
        rows = layer_dims[i]
        cols = layer_dims[i - 1]

        W.append(np.random.randn(rows, cols)/10)
        # W.append(np.random.randn(rows, cols) * np.sqrt(2 / cols))
        b.append(np.zeros(rows).reshape(rows, 1))

        # if i == 1:
        #     W.append(np.random.normal(0, 1 / np.sqrt(cols), [rows, cols]))
        # else:
        #     W.append(np.random.normal(0, 1, [rows, cols]))

        # W.append(np.random.normal(0, 10 / np.sqrt(cols), [rows, cols]))
        # b.append(np.zeros(rows).reshape(rows, 1))

    return {'W': W, 'b': b}


def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation
    :param A: activations of the previous layer
    :param W: the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    :param b: the bias vector of the current layer (of shape [size of current layer, 1])
    :return: Z – the linear component of the activation function (i.e., the value before applying the non-linear function)
            linear_cache – a dictionary containing A, W, b (stored for making the backpropagation easier to compute)
    """
    return {'Z': np.matmul(W, A) + b, 'linear_cache': {'A': A, 'W': W, 'b': b}}


def softmax(Z):
    """
    Implements the softmax activation
    :param Z: the linear component of the activation function
    :return: A – the activations of the layer.
            activation_cache – returns Z, which will be useful for the backpropagation
    """
    exp = np.exp(Z - np.max(Z))
    A = exp / np.sum(exp, axis=0)
    return {'A': A, 'activation_cache': Z}

    # A = []
    # for idx in range(0, Z.shape[1]):
    #     curr = Z[:, idx]
    #     e = np.exp(curr)
    #     a = e / np.sum(e)
    #     A.append(a)
    # return {'A': np.array(A).T, 'activation_cache': Z}

    # A = scipy_softmax(Z)
    # return {'A': A, 'activation_cache': Z}

def relu(Z):
    """
    Implements the relu activation
    :param Z: the linear component of the activation function
    :return: A – the activations of the layer.
            activation_cache – returns Z, which will be useful for the backpropagation
    """
    return {'A': np.abs(Z) * (Z > 0), 'activation_cache': Z}


def linear_activation_forward(A_prev, W, B, activation):
    """
    Implements the forward propagation for the LINEAR->ACTIVATION layer
    :param A_prev: activations of the previous layer
    :param W: the weights matrix of the current layer
    :param B: the bias vector of the current layer
    :param activation: the activation function to be used (a string, either “softmax” or “relu”)
    :return: A – the activations of the current layer.
            cache – a joint dictionary containing both linear_cache and activation_cache
    """
    l_f = linear_forward(A_prev, W, B)
    lin_cache, Z = l_f['linear_cache'], l_f['Z']
    cache = [lin_cache, Z]
    if activation == 'relu':
        act = relu(Z)
    elif activation == 'softmax':
        act = softmax(Z)
    else:
        raise ValueError('activation must be either "relu" or "softmax"')
    return {'A': act['A'], 'cache': cache}


def apply_batchnorm(A, epsilon=0.000001):
    """
    Performs batchnorm on the received activation values of a given layer.
    :param A: the activation values of a given layer
    :param epsilon: to avoid division by 0
    :return: NA - the normalized activation values, based on the formula learned in class
    """
    return (A - np.mean(A)) / np.sqrt(np.var(A) + epsilon)


def L_model_forward(X, parameters, use_batchnorm):
    """
    Implements forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation
    :param X: the data, numpy array of shape (input size, number of examples)
    :param parameters: the initialized W and b parameters of each layer
    :param use_batchnorm: a boolean flag used to determine whether to apply batchnorm after the activation
            (note that this option needs to be set to “false” in Section 3 and “true” in Section 4).
    :return: AL – the last post-activation value.
            caches – a list of all the cache objects generated by the linear_forward function
    """
    W, b = parameters['W'], parameters['b']
    L = len(b)
    activations = ['relu'] * (L - 1) + ['softmax']
    caches = []
    A = X
    for i in range(L):
        layer_output = linear_activation_forward(A, W[i], b[i], activations[i])
        A = layer_output['A']
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(layer_output['cache'])
    return A, caches


def compute_cost(AL, Y):
    """
    Implements the cost function defined by equation. The requested cost function is categorical cross-entropy loss
    :param AL: probability vector corresponding to your label predictions, shape (num_of_classes, number of examples)
    :param Y: the labels vector (i.e. the ground truth)
    :return: cost – the cross-entropy cost
    """
    eps = 0.00000000001
    loss = -(1 / len(Y)) * (Y * np.log(AL + eps) + (1 - Y) * np.log(1 - AL + eps))
    return np.sum(loss * Y, axis=0)  # keep only loss from positive class
