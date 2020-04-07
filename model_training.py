import numpy as np

from backward_propagation import L_model_backward, update_parameters
from forward_propagation import initialize_parameters, L_model_forward, compute_cost


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size,
                  use_batchnorm=False, save_cost_at=100, acc_tier_size=0.005, early_stop_spree=100,validation_not_improving = 100):
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
    :param early_stop_spree: length of spree of epochs in same acc_tier to stop training at
    :return: parameters – the parameters learnt by the system during the training
            costs – the values of the cost function (calculated by the compute_cost function)
            One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values)
    """
    X_train = X["train"]
    X_validation = X["validation"]
    Y_train = Y["train"]
    Y_validation = Y["validation"]

    parameters = initialize_parameters(layers_dims)

    costs = []
    instance_count = X_train.shape[1]
    batches = int(instance_count / batch_size)
    best_accuracy = 0
    iteration = 0
    count_no_cahnge = 0
    validation_acc = -1
    while iteration != num_iterations:  # implemented as while to allow infinite max iterations
        print(iteration)
        cost = None
        if iteration % save_cost_at == 0 or iteration == num_iterations - 1:  # save cost if modulo or last epoch
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
        if cost is not None:
            costs.append(cost)


        # Validation
        new_validation_acc = Predict(X_validation,Y_validation,parameters)

        if new_validation_acc <= validation_acc:
            print("new %s old %s" % (new_validation_acc,validation_acc))
            count_no_cahnge = count_no_cahnge + 1
            if count_no_cahnge == validation_not_improving:
                print("here1")
                break
        else:
            print("new %s old %s - improved!" % (new_validation_acc,validation_acc))
            count_no_cahnge = 0
        validation_acc = new_validation_acc

        if iteration % early_stop_spree == 0:  # check for early stop
            accuracy = Predict(X_train, Y_train, parameters)
            if accuracy < best_accuracy + acc_tier_size:
                print("here2")
                break  # end training
            best_accuracy = accuracy
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

"""
instances, input_dim, output_dim = 10, 6, 2
x = np.random.randn(input_dim, instances)
y = np.zeros([output_dim, instances])
y[-1] = np.ones(instances)
layers_dims = [input_dim, input_dim - 2, output_dim]
X = {"train": x, "validation": x}
Y = {"train": y, "validation": y}
parameters, costs = L_layer_model(X, Y ,layers_dims, 0.05, 200, int(instances/2))
accuracy = Predict(x, y, parameters)

print(x)
print(y)
print(y.shape)
print(layers_dims)

print('accuracy = %.4f' % accuracy)
print(costs)
"""