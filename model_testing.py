import csv

import keras
import numpy as np
import sklearn.model_selection
from model_training import Predict, L_layer_model


def get_data_set():
    """
    Returns the MNIST dataSet uniting the train and test it returns
    :return: The data set (MNIST)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2])) / 255

    return x, y


def divide_dataset(x, y, num_of_classes):
    """
    This function will split the dataset into train validation and test sets
    :param x: The features (instances)
    :param y: The true nlabels of the instances (classes)
    :return: train validation and test sets
    """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.2)
    X_train = X_train.T
    y_train = convert_to_matrix(y_train, num_of_classes)
    X_val = X_val.T
    y_val = convert_to_matrix(y_val, num_of_classes)
    X_test = X_test.T
    y_test = convert_to_matrix(y_test, num_of_classes)

    return X_train, y_train, X_val, y_val, X_test, y_test


def convert_to_matrix(y, num_of_class):
    """
    Transforms a vector y with num_of_class different values into a [num_of_class, n] matrix
    :param y: true labels
    :param num_of_class: number of different classes
    :return: [num_of_class, n] matrix
    """
    new_y = np.zeros((num_of_class, y.shape[0]))
    for i in range(len(y)):
        new_y[y[i]][i] = 1
    return new_y


# main
x, y = get_data_set()
X_train, y_train, X_val, y_val, X_test, y_test = divide_dataset(x, y, 10)
X = [X_train, X_val]
Y = [y_train,  y_val]
layers_dims = [x.shape[1], 20, 7, 5, 10]
use_batchnorm = True
print('use_batchnorm = %s' % use_batchnorm)
parameters, costs = L_layer_model(X, Y, layers_dims, 0.009, -1, 2000, use_batchnorm=use_batchnorm)
train_acc = Predict(X_train, y_train, parameters, use_batchnorm)
test_acc = Predict(X_test, y_test, parameters, use_batchnorm)
print('train_acc = %.4f' % train_acc)
print('test_acc = %.4f' % test_acc)

with open('log.csv', 'a', newline='') as log:
    log_writer = csv.writer(log)
    log_writer.writerow([])
    log_writer.writerow(['train_acc', 'test_acc'])
    log_writer.writerow([train_acc, test_acc])

