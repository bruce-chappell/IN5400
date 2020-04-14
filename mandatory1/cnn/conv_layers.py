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

"""Implementation of convolution forward and backward pass"""

import numpy as np

def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_alyer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    # TODO: Task 2.1
    #output_layer = None # Should have shape (batch_size, num_filters, height_y, width_y)

    (batch_size, channels_x, height_x, width_x) = input_layer.shape
    #[i,         k,          height_x, width_x]
    
    (num_filters, channels_w, height_w, width_w) = weight.shape
    #[j,          k,          height_w, width_w]
    
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")
    
    #output image considering stride
    H_y = int(1 + (height_x + 2*pad_size - height_w)/stride)
    W_y = int(1 + (width_x + 2*pad_size - height_w)/stride)
    output_layer = np.zeros((batch_size, num_filters, H_y, W_y))
    #                        [i,         j,           h,   w] 
    
    
    for i in range(batch_size):
        for j in range(num_filters): 
            #sum over channels-> normal conv for each channel-> week1 ex
            for k in range(channels_x):
                inp_pad = np.pad(input_layer[i,k,:,:], pad_size, 'constant', constant_values=0)
                for h in range(H_y):
                    for w in range(W_y):
                        output_layer[i,j,h,w] += np.sum(inp_pad[h*stride:h*stride+height_w, w*stride:w*stride+width_w]
                                                        *weight[j,k,:,:])
            #bias for each filter
            output_layer[i,j,:,:] += bias[j]
                        
    return output_layer


def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size=1):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    # TODO: Task 2.2
    #input_layer_gradient, weight_gradient, bias_gradient = None, None, None

    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape

    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")
    
    #grads must be same size as their variable
    inp_grad = np.zeros_like(input_layer)
    b_grad = np.zeros(num_filters)
    w_grad = np.zeros_like(weight)
    
    for i in range(batch_size):
        for j in range(num_filters):
            for k in range(channels_x):
                
                #pad so dimensions line up for "backwards convolving"
                inp_pad = np.pad(input_layer[i,k,:,:], pad_size, 'constant', constant_values=0)
                out_grad_pad = np.pad(output_layer_gradient[i,j,:,:], pad_size, 'constant', constant_values=0)
                #180 flip
                weight_flip = weight[j,k,-1::-1,-1::-1]
                # convolve output grad over padded input
                for m in range(height_w):
                    for n in range(width_w):
                        w_grad[j,k,m,n] += np.sum(inp_pad[m:m+height_y, n:n+width_y] * output_layer_gradient[i,j,:,:])
                #convolve flipped weights over padded output grad
                for s in range(height_x):
                    for r in range(width_x):
                        inp_grad[i,k,s,r] += np.sum(out_grad_pad[s:s+height_w, r:r+width_w] * weight_flip)
            
            b_grad[j] += np.sum(output_layer_gradient[i,j,:,:])
    
    return inp_grad, w_grad, b_grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
