import numpy as np

from backward_propagation import L_model_backward, update_parameters
from forward_propagation import initialize_parameters, L_model_forward, compute_cost


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size,
                  use_batchnorm=False, save_cost_at=100, acc_tier=0.005, same_acc_tier_spree=100):
    """
    Implements a L-layer neural network. All layers but the last should have the ReLU activation function,
    and the final layer will apply the softmax activation function. The size of the output layer should be equal to
    the number of labels in the data. Please select a batch size that enables your code to run well (i.e. no memory
    overflows while still running relatively fast).
    :param X: the input data, a numpy array of shape (height*width , number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate: lr of the model
    :param num_iterations: epochs
    :param batch_size: the number of examples in a single training batch
    :param use_batchnorm: to batchnorm the output of layers
    :param save_cost_at: iteration modulo in which to save cost of output
    :param acc_tier: accuracy tier size for knowing when to stop training
    :param same_acc_tier_spree: length of spree of epochs in same acc_tier to stop training at
    :return: parameters – the parameters learnt by the system during the training
            costs – the values of the cost function (calculated by the compute_cost function)
            One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values)
    """
    parameters = initialize_parameters(layers_dims)
    costs = []
    instance_count = X.shape[1]
    batches = int(instance_count / batch_size)
    for iteration in range(len(num_iterations)):
        cost = None
        if iteration % save_cost_at == 0 or iteration == len(num_iterations) - 1:
            cost = np.empty(0)
        for batch in range(batches + 1):
            batch_start = batch * batch_size
            if batch_start == instance_count:
                continue  # in case the len of train set is a multiple of batch size
            batch_end = min((batch + 1) * batch_size, instance_count)  # avoid index out of bounds
            X_batch = X[:, batch_start:batch_end]
            Y_batch = Y[:, batch_start:batch_end]
            AL, caches = L_model_forward(X_batch, parameters, use_batchnorm)
            if cost is not None:
                cost = np.append(cost, compute_cost(AL, Y_batch))
            grads = L_model_backward(AL, Y_batch, caches)
            update_parameters(parameters, grads, learning_rate)
        if cost is not None:
            costs.append(cost)
    return parameters, costs


def accuracy(predictions, Y):
    """
    Compute accuracy of predictions
    :param predictions:
    :param Y:
    :return:
    """


def Predict(X, Y, parameters):
    """
    The function receives an input data and the true labels and calculates the accuracy of the trained nn on the data.
    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: python dictionary containing the DNN architecture’s parameters
    :return: accuracy – the accuracy measure of the neural net on the provided data
    """
    predictions = np.round(L_model_forward(X, parameters, False)[0])
    accuracy = np.mean(np.equal(predictions, Y).astype(int))
