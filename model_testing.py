import keras
import numpy as np
import sklearn.model_selection.train_test_split
from model_training import Predict, L_layer_model
def get_data_set():
    """
    This function will return the dataSet
    :return: The data set (MNIST)
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    print(x.shape)
    return x,y

def divide_data_set_into_train_test_validation_set(x,y):
    """
    This function will split the dataSet into train validation and test sets

    :param x: The features (instances)
    :param y: The true nlabels of the instances (classes)
    :return: train validation and test sets
    """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X_train, y_train, test_size=0.2)
    return X_train,y_train,X_val,y_val,X_test,y_test


def test_model():
    x,y = get_data_set()
    X_train, y_train, X_val, y_val, X_test, y_test = divide_data_set_into_train_test_validation_set(x,y)
    X = {"train" : X_train,"validation" : X_val}
    Y = {"train" : y_train,"validation" : y_val}
    # def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size,
    # use_batchnorm=False, save_cost_at=100, acc_tier_size=0.005, early_stop_spree=100):
    #parameters, costs = L_layer_model(X,Y)


get_data_set()