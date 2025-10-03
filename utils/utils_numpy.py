import numpy as np

def softmax(x, axis=-1):

    #subtract max for stability
    exp_x = np.exp(x-np.max(x, axis=axis, keepdims=True))
    
    return exp_x/np.sum(exp_x, axis=axis, keepdims=True)