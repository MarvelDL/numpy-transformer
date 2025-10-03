import numpy as np
from models.encoder import Encoder

config  =  {
    "d_model": 512,
    "num_heads": 8,
    "d_ff":2048,
    "num_layers": 6,
    "vocab_size": 10000,
    "max_len": 500
}
 
x = np.array([[1, 2, 3, 4, 0],
              [6, 7, 0, 0, 0]]) 

encodr = Encoder(**config)

out, attn_weights = encodr.forward(x)

print("Input:", x)
print("\nOutput:\n", out)
print("\nAttention Weights size:\n", len(attn_weights))
