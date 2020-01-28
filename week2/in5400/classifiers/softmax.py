import numpy as np
from random import shuffle

def softmax(x):
    x = x - x.max(axis = 1, keepdims = True)
    t = x - np.log(np.sum(np.exp(x), axis = 1, keepdims = True))
    return np.exp(t)

def one_hot(x):
    n_inputs = len(x)
    n_categories = np.max(x) + 1
    vec = np.zeros((n_inputs, n_categories))
    vec[range(n_inputs),x] = 1
    return vec
        

def softmax_loss_naive(W, X, y):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    n_inputs = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    z = X @ W
    a = softmax(z)
    y_true = one_hot(y)
    hold = (a - y_true)
    dW = np.dot(X.T, hold)
    
    temp = np.log(a)
    loss = -1/n_inputs * np.sum(np.sum(temp*y_true, axis=1))
    
    #loss=[]
    #dw = []
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    n_inputs = X.shape[0]
    z = X @ W
    a = softmax(z)
    y_true = one_hot(y)
    hold = (a - y_true)
    dW = np.dot(X.T, hold)
    
    temp = np.log(a)
    loss = -1/n_inputs * np.sum(np.sum(temp*y_true, axis=1))
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

