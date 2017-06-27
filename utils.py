# coding: utf-8
import tensorflow as tf
import numpy as np
import pickle

def create_weight_variable(shape, name="W"):
    """
    Create a tensorflow variable, initiale with a normal law.

    Args:
        shape (list or tupl): shape of the variable (None for undefined dimension)
        name (str): Name of the variable in the tensorflow graph

    Returns:
        tf.Variable: the tensorflow variable created
    """

    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
        
def create_bias_variable(shape, name="b"):
    """
    Create a tensorflow variable, initiale with a constant.

    Args:
        shape (list or tupl): shape of the variable (None for undefined dimension)
        name (str): Name of the variable in the tensorflow graph

    Returns:
        tf.Variable: the tensorflow variable created
    """
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def saver(params, sess, save_path):
    """
    Custom saver for tensorflow model

    Args:
        params (dict):  dictionnaire containing "weights" and "biases", in which are stocked the tensorflow 
                variables corresponding to the weights and the biases of the model.
        sess (tf.Session): Current tensorflow Session
        save_path (str): The path where the model has to be saved
    """
    weights = params["weights"]
    biases = params["biases"]
    weights_saver = {key: sess.run(weights[key]) for key in weights}
    biases_saver = {key: sess.run(biases[key]) for key in biases}
    dico_saver = {"weights": weights_saver, "biases": biases_saver}
    with open(save_path, "w") as fp:
        pickle.dump(dico_saver, fp)

def loader(params, sess, save_path):
    """
    Custom loader for tensorflow model. Careful, the model must be created, it will only assign values
    to existing variable.

    Args:
        params (dict):  dictionnaire containing "weights" and "biases", in which are stocked the tensorflow 
                variables corresponding to the weights and the biases of the model.
        sess (tf.Session): Current tensorflow Session
        save_path (str): The path where the model is saved
    """
    weights = params["weights"]
    biases = params["biases"]
    with open(save_path, "r") as fp:
        dico_saver = pickle.load(fp)
    weights_saver = dico_saver["weights"]
    biases_saver = dico_saver["biases"]
    for key in weights:
        if key in weights_saver:
            sess.run(weights[key].assign(weights_saver[key]))
    for key in biases:
        if key in biases_saver:
            sess.run(biases[key].assign(biases_saver[key]))

def compute_accuracy(nn, X, y, sess, dropout=None):
    """
    Compute the accuracy for a binary classification

    Args:
        nn (model): model which must have out_tensor, input_tensor and dropout attribute
        X (np.array): data on which the accuracy is computed
        y (np.array): labels of the data
        sess (tf.Session): current running Session
        dropout (float): keeping probability for the input. Set to None if there is no dropout.

    Returns:
        float: The computed accuracy
    """
    if dropout is None:
        feed_dict = {nn.input_tensor: X}
    else:
        feed_dict = {nn.input_tensor: X, nn.dropout: dropout}
    return np.mean((sess.run(nn.out_tensor, feed_dict=feed_dict)>0.5) == y)

def compute_reconstruction_error(ae, X, sess, dropout=False):
    """
    Compute the reconstruction error for a autoencoder.

    Args:
        ae (model): auto_encoder which have loss, input_tensor and can have dropout attribute.
        X (np.array): data on wich the error is calculted
        sess (tf.Session): current running Session
        dropout (Boolean): Is the autoencoder denoising ?

    Returns:
        float: the computed error
    """
    if dropout:
        feed_dict = {ae.input_tensor: X, ae.dropout: 1.}
    else:
        feed_dict = {ae.input_tensor: X}
    return sess.run(ae.loss, feed_dict=feed_dict)
