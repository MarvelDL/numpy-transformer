import numpy as np
from utils.utils_numpy import softmax

def scaled_dot_product_attention(Q, K, V, mask=None):

    # d_k = dimension of queries and keys
    d_k = Q.shape[-1]

    # Raw attention scores (QK^T)
    scores = np.matmul(Q, K.transpose(0,2,1))

    #Scaling by sqrt(d_k)
    scores = scores/np.sqrt(d_k)

    # Apply mask (set masked positions to -1e9 so that softmax ~ 0)
    if mask is not None:
        scores = np.where(mask==0, -1e9, scores)

    # Softmax across last axis (keys)
    attention_weights  = softmax(scores, axis=-1)

    # weighted sum of values
    output = np.matmul(attention_weights, V)


    return output, attention_weights
    
