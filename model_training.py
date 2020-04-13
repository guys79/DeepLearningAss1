import csv
from random import randrange, sample
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


def get_selected_neurons_index(selected):
    indexes = []
    for i in range(len(selected)):
        if selected[i]:
            indexes.append(i)
    if len(indexes) == 0:
        indexes.append(randrange(len(selected)))
        selected[indexes[0]] = True
    return np.array(indexes)

def get_parameters_for_training(keep_prob, parameters, layers_dims):
    """
    This function will return teh parameters that needs to be trained
    :param keep_prob: The probability for a neuron to not drop out
    :param parameters: The model's parameters
    :param layers_dims: The number of neurons in each layer in the model
    :return: The parameters to train
    """
    if keep_prob < 1:
        W_parameters = parameters['W']
        new_W_parameters = []
        b_parameters = parameters['b']
        new_b_parameters = []
        chosen_neurons = []

        # For each layer
        for index in range(1, len(layers_dims) - 1):
            neurons_to_train = []
            #neurons_to_train = np.random.rand((layers_dims[index])) < keep_prob
            indices = sample(range(layers_dims[index]),int(keep_prob*layers_dims[index]))
            #print("%d/%d" %(len(indices), layers_dims[index]))
            for i in range(layers_dims[index]):
                if i in indices:
                    neurons_to_train.append(True)
                else:
                    neurons_to_train.append(False)
            chosen_neurons.append(neurons_to_train)
            w_for_layer = W_parameters[index - 1]
            b_for_layer = b_parameters[index - 1]

            if index == 1:
                selected_previous = range(layers_dims[0])
            else:
                selected_previous = get_selected_neurons_index(chosen_neurons[len(chosen_neurons) - 2])
            selected_now = get_selected_neurons_index(chosen_neurons[len(chosen_neurons) - 1])
          #  print("amount chosen to train %d out if %d" % (len(selected_now), len(neurons_to_train)))
            new_w_for_layer = w_for_layer[selected_now[:, None], selected_previous]
            new_b_for_layer = b_for_layer[selected_now]
           # print(new_w_for_layer.shape)
           # print(new_b_for_layer.shape)
           # print("guy")

            new_W_parameters.append(new_w_for_layer)
            new_b_parameters.append(new_b_for_layer)

        new_W_parameters.append(W_parameters[len(layers_dims) - 2][:, selected_now])
        new_b_parameters.append(b_parameters[len(layers_dims) - 2])
        parameters_for_training = {'W': new_W_parameters, 'b': new_b_parameters}

    else:
        parameters_for_training = parameters
        chosen_neurons = []
        for i in range(1,len(layers_dims)-1):
            chosen_in_layer = [True]*layers_dims[i]
            chosen_neurons.append(chosen_in_layer)
    return parameters_for_training,chosen_neurons

def merge_results(parameters, parameters_for_training, chosen_neurons,keep_prob):
    """
    This function will merge the parameters after being trained with the model's parameters
    :param parameters: The model's parameters
    :param parameters_for_training: The parameters that have been trained
    :param chosen_neurons: The neurons that didn't dropout
    :param keep_prob: The probability for a neuron to not drop out
    :return: The model's parameters
    """
    #If there is no Dropout
    if keep_prob == 1:
        parameters = parameters_for_training
        return parameters

    W_parameters = parameters['W']
    trained_W_parameters = parameters_for_training['W']
    b_parameters = parameters['b']
    trained_b_parameters = parameters_for_training['b']

    # First layer
    chosen_neurons_in_layer = chosen_neurons[0]
    W_parameters_in_layer = W_parameters[0]
    trained_W_parameters_in_layer = trained_W_parameters[0]
    b_parameters_in_layer = b_parameters[0]
    trained_b_parameters_in_layer = trained_b_parameters[0]
    trained_idx = 0
    for neuron_idx in range(len(chosen_neurons_in_layer)):
        if chosen_neurons_in_layer[neuron_idx]:
            W_parameters_in_layer[neuron_idx] = trained_W_parameters_in_layer[trained_idx]
            b_parameters_in_layer[neuron_idx] = trained_b_parameters_in_layer[trained_idx]
            trained_idx+=1

    # For each layer in the middle

    for neuron_layer in range(1,len(chosen_neurons)):
        W_parameters_in_layer = W_parameters[neuron_layer]
        trained_W_parameters_in_layer = trained_W_parameters[neuron_layer]
        b_parameters_in_layer = b_parameters[neuron_layer]
        trained_b_parameters_in_layer = trained_b_parameters[neuron_layer]

        rows = chosen_neurons[neuron_layer]
        cols = chosen_neurons[neuron_layer-1]

        trained_row_idx = 0
        for row in range(len(rows)):
            trained_col_idx = 0
            if rows[row]:
                for col in range(len(cols)):
                    if cols[col]:
                        W_parameters_in_layer[row][col] = trained_W_parameters_in_layer[trained_row_idx][trained_col_idx]
                        trained_col_idx+=1
                b_parameters_in_layer[row] = trained_b_parameters_in_layer[trained_row_idx]
                trained_row_idx+=1

    #Last layer
    W_parameters_in_layer = W_parameters[len(W_parameters)-1]
    trained_W_parameters_in_layer = trained_W_parameters[len(trained_W_parameters)-1]

    trained_col_idx = 0
    for col_idx in range(len(chosen_neurons[len(chosen_neurons)-1])):
        if chosen_neurons[len(chosen_neurons)-1][col_idx]:
            for row_idx in range(len(W_parameters_in_layer)):
                W_parameters_in_layer[row_idx][col_idx] = trained_W_parameters_in_layer[row_idx][trained_col_idx]
            trained_col_idx+=1


    b_parameters[len(b_parameters) - 1] = trained_b_parameters[len(trained_b_parameters) - 1]
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm, iters_in_round=100,keep_prob =1):
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
    :param keep_prob: The probability for a neuron to not dropout (1 if there is no drop out)
    :return:
    """
    X_train, X_val = X
    Y_train, Y_val = Y
    batches = get_batches(batch_size, X_train, Y_train)
    parameters = initialize_parameters(layers_dims)
    best_val_cost = -1
    iters_with_no_improvement = 0
    costs = []
    epoch = 1
    iteration = 0
    done = False
    """
    layers_dims_shape = (layers_dims[0])
    for index in range(1,len(layers_dims)):
        layers_dims_shape+=(layers_dims[index])
    """

    with open('log.csv', 'w', newline='') as log:
        log_writer = csv.writer(log)
        log_writer.writerow(['epoch', 'iteration', 'val_cost', 'val_acc'])
    while not done:

#        if epoch % change_neurons_freq == 1:

        for X_batch, Y_batch in batches:
            iteration += 1
            parameters_for_training, chosen_neurons = get_parameters_for_training(keep_prob, parameters, layers_dims)
            AL, caches = L_model_forward(X_batch, parameters_for_training, use_batchnorm, keep_prob = keep_prob)
            grads = L_model_backward(AL, Y_batch, caches)
            update_parameters(parameters_for_training, grads, learning_rate)
            parameters = merge_results(parameters, parameters_for_training, chosen_neurons, keep_prob)
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
