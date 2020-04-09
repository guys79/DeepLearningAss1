import csv

import numpy as np

from backward_propagation import L_model_backward, update_parameters
from forward_propagation import initialize_parameters, L_model_forward, compute_cost


def get_batches(batch_size, x_train, y_train):
    """
    Get the list of batches for training
    :param batch_num: num of batches
    :param batch_size: size of batch
    :param x_train: train instances
    :param y_train: train labels
    :return: list of batches
    """
    instance_count = y_train.shape[1]
    batch_num = int(instance_count / batch_size)
    batches = []
    for batch in range(batch_num + 1):
        batch_start = batch * batch_size
        if batch_start == instance_count:
            break  # in case the len of train set is a multiple of batch size
        batch_end = min((batch + 1) * batch_size, instance_count)  # avoid index out of bounds
        x_batch = x_train[:, batch_start:batch_end]
        y_batch = y_train[:, batch_start:batch_end]
        batches.append([x_batch, y_batch])
    return batches


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm, iters_in_round=100):
    """
    Implements a L-layer neural network. All layers but the last should have the ReLU activation function,
    and the final layer will apply the softmax activation function. The size of the output layer should be equal to
    the number of labels in the data. Please select a batch size that enables your code to run well (i.e. no memory
    overflows while still running relatively fast).
    :param X: the input data, a numpy array of shape (height*width , number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate: lr of the model
    :param num_iterations: max epochs
    :param batch_size: the number of examples in a single training batch
    :param use_batchnorm: to batchnorm the output of layers
    :param iters_in_round: length of spree of epochs in same acc_tier to stop training at
    :return: parameters – the parameters learnt by the system during the training
            costs – the values of the cost function (calculated by the compute_cost function)
            One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values)
    """
    X_train, X_val = X
    Y_train, Y_val = Y
    batches = get_batches(batch_size, X_train, Y_train)
    parameters = initialize_parameters(layers_dims)
    # best_val_cost_prev_round = -1
    best_val_cost = -1
    iters_with_no_improvement = 0
    costs = []
    epoch = 1
    iteration = 0
    done = False
    with open('log.csv', 'w', newline='') as log:
        log_writer = csv.writer(log)
        log_writer.writerow(['epoch', 'iteration', 'val_cost', 'val_acc'])
    while not done:
        for X_batch, Y_batch in batches:
            iteration += 1
            AL, caches = L_model_forward(X_batch, parameters, use_batchnorm)
            grads = L_model_backward(AL, Y_batch, caches)
            update_parameters(parameters, grads, learning_rate)
            AL_val = L_model_forward(X_val, parameters, use_batchnorm)[0]
            val_cost = compute_cost(AL_val, Y_val)
            if best_val_cost == -1 or val_cost < best_val_cost:
                best_val_cost = val_cost
                iters_with_no_improvement = 0
            else:
                iters_with_no_improvement += 1

            # do every iterations_in_round iterations
            if iteration % iters_in_round == 0:
                costs.append(val_cost)
                val_acc = Predict(X_val, Y_val, parameters, use_batchnorm)
                print('epoch=%d iter=%d val_cost=%.4f val_acc=%.4f'
                      % (epoch, iteration, best_val_cost, val_acc))
                with open('log.csv', 'a', newline='') as log:
                    log_writer = csv.writer(log)
                    log_writer.writerow([epoch, iteration, best_val_cost, val_acc])

            if iteration == num_iterations or iters_with_no_improvement == iters_in_round:
                done = True
                break  # end training
        epoch += 1
    print('epochs=%d iters=%d' % (epoch, iteration))
    return parameters, costs


def Predict(X, Y, parameters, use_batchnorm):
    """
    The function receives an input data and the true labels and calculates the accuracy of the trained nn on the data.
    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: python dictionary containing the DNN architecture’s parameters
    :return: accuracy – the accuracy measure of the neural net on the provided data
    """
    AL = L_model_forward(X, parameters, use_batchnorm)[0]
    predictions = (AL == np.amax(AL, axis=0)).astype(int)
    return np.sum(predictions * Y) / Y.shape[1]
