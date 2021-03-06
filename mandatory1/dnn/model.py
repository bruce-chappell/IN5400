#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1.1
    params={}
    lengths = conf['layer_dimensions']
    weights = [np.random.normal(0,np.sqrt(2/d1),size=[d1,d2]) for d1, d2 in zip(lengths[:-1], lengths[1:])]
    bias = [np.zeros((d,1)) for d in lengths[1:]]
    for i in range(len(weights)):
        params[f'W_{i+1}'] = weights[i]
        params[f'b_{i+1}'] = bias[i]

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 a)
    if activation_function == 'relu':
        return np.maximum(Z,0)
    if activation_function == 'softmax':
        return softmax(Z)

    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.2 b)

    Z = Z - Z.max(axis = 0, keepdims = True)
    t = Z - np.log(np.sum(np.exp(Z), axis = 0, keepdims = True))
    return np.exp(t)


def forward(conf, X_batch, params, is_training = True):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 1.2 c
    layers = conf['layer_dimensions']
    lay_num = len(layers)
    activations = {}
    features = {}
    #create a dictionary of the activations and output function
    for i in range(lay_num-2):
        activations[f'layer_{i+1}'] = 'relu'
    activations[f'layer_{lay_num-1}'] = 'softmax'
        
    features['A_0'] = X_batch
    
    for i in range(lay_num-1):
        features[f'Z_{i+1}'] = np.dot(params[f'W_{i+1}'].T, features[f'A_{i}']) + params[f'b_{i+1}']
        features[f'A_{i+1}'] = activation(features[f'Z_{i+1}'], activations[f'layer_{i+1}'])
    Y_proposed = features[f'A_{lay_num-1}']

    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 1.3
    temp = np.log(Y_proposed)
    m = Y_proposed.shape[1]
    classes = Y_proposed.shape[0]
    cost = -1/m * np.sum(np.sum(temp*Y_reference, axis=0))
    
    idx = np.argmax(Y_proposed, axis = 0)
    pred = np.zeros_like(Y_proposed)   
    num_correct = 0
    
    for i in range(m):
        pred[idx[i],i] = 1
        temp = np.array_equal(pred[:,i], Y_reference[:,i])
        num_correct += int(temp)

    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 1.4 a)
    if activation_function == 'relu':
        return 1 * (Z >= 0)
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 1.4 b)
    grad_params = {}
    layers = conf['layer_dimensions']
    act = conf['activation_function']
    scale = 1/Y_proposed.shape[1]
    L = len(layers)-1
    delta = Y_proposed - Y_reference
    grad_params[f'grad_W_{L}'] = scale * np.dot(features[f'A_{L-1}'], delta.T)
    grad_params[f'grad_b_{L}'] = scale * np.sum(delta, axis = 1, keepdims = True)
    
    for l in reversed(range(1,L)):
        delta = np.dot(params[f'W_{l+1}'], delta) * activation_derivative(features[f'Z_{l}'], act)
        grad_params[f'grad_W_{l}'] = scale * np.dot(features[f'A_{l-1}'], delta.T)
        grad_params[f'grad_b_{l}'] = scale * np.sum(delta, axis = 1, keepdims =True)
    
    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    # TODO: Task 1.5
    lrn = conf['learning_rate']
    L = int(len(params) / 2)
    updated_params = {}
    for l in reversed(range(1,L+1)):
        updated_params[f'W_{l}'] = params[f'W_{l}'] - lrn * grad_params[f'grad_W_{l}']
        updated_params[f'b_{l}'] = params[f'b_{l}'] - lrn * grad_params[f'grad_b_{l}']

    return updated_params
