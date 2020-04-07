import numpy as np

from backward_propagation import L_model_backward, update_parameters
from forward_propagation import initialize_parameters, L_model_forward, compute_cost


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size,
                  use_batchnorm=False, iterations_to_check=100, acc_tier_size=0.005):
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
    :param save_cost_at: iteration modulo in which to save cost of output
    :param acc_tier_size: accuracy tier size for knowing when to stop training
    :param iterations_to_check: length of spree of epochs in same acc_tier to stop training at
    :return: parameters – the parameters learnt by the system during the training
            costs – the values of the cost function (calculated by the compute_cost function)
            One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values)
    """
    X_train = X["train"]
    X_val = X["validation"]
    Y_train = Y["train"]
    Y_val = Y["validation"]
    parameters = initialize_parameters(layers_dims)
    costs = []
    instance_count = X_train.shape[1]
    batches = int(instance_count / batch_size)
    best_val_acc = -1
    iteration = 0
    while iteration != num_iterations:  # implemented as while to allow infinite max iterations
        cost = None
        if iteration % iterations_to_check == 0 or iteration == num_iterations - 1:  # save cost if modulo or last epoch
            cost = np.empty(0)
        for batch in range(batches + 1):
            batch_start = batch * batch_size
            if batch_start == instance_count:
                continue  # in case the len of train set is a multiple of batch size
            batch_end = min((batch + 1) * batch_size, instance_count)  # avoid index out of bounds
            X_batch = X_train[:, batch_start:batch_end]
            Y_batch = Y_train[:, batch_start:batch_end]
            AL, caches = L_model_forward(X_batch, parameters, use_batchnorm)
            if cost is not None:
                cost = np.append(cost, compute_cost(AL, Y_batch))
            grads = L_model_backward(AL, Y_batch, caches)
            update_parameters(parameters, grads, learning_rate)
        # if cost is not None:
        #     costs.append(cost)

        if iteration % iterations_to_check == 0:  # check for early stop
            costs.append(cost)
            train_acc = Predict(X_train, Y_train, parameters)
            val_acc = Predict(X_val, Y_val, parameters)
            print('%d train_cost=%d train_acc=%.4f val_acc=%.4f' % (iteration, np.sum(cost), train_acc, val_acc))
            if val_acc < best_val_acc + acc_tier_size:
                break  # end training
            best_val_acc = val_acc

        iteration += 1
    return parameters, costs


def Predict(X, Y, parameters):
    """
    The function receives an input data and the true labels and calculates the accuracy of the trained nn on the data.
    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: python dictionary containing the DNN architecture’s parameters
    :return: accuracy – the accuracy measure of the neural net on the provided data
    """
    AL = L_model_forward(X, parameters, False)[0]
    predictions = (AL == np.amax(AL, axis=0)).astype(int)
    return np.sum(predictions * Y) / Y.shape[1]
