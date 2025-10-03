import numpy as np

NEG_INF = -1e9

def create_padding_mask(seq, pad_token=0):

    mask = (seq == pad_token).astype(np.float32)
    mask =  mask[:, np.newaxis, np.newaxis, :]
    return mask * NEG_INF

def create_look_ahead_mask(size):

    allowed = np.tril(np.ones((size, size), dtype=np.float32))
    blocked = (1.0 - allowed) * NEG_INF
    return blocked[np.newaxis, np.newaxis, :, :]

def combine_masks(look_ahead_mask, padding_mask):

    if padding_mask is None:
        return look_ahead_mask
    if look_ahead_mask is None:
        return padding_mask
    
    return np.maximum(look_ahead_mask, padding_mask)