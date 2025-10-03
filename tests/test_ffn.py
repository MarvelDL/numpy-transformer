import numpy as np
from feed_forward.ffn import FeedForwardNetwork

batch_size, seq_len, d_ff, d_model = 1, 2, 8, 4

x = np.arange(1, batch_size*seq_len*d_model + 1).reshape(batch_size, seq_len, d_model).astype(float)

ffn = FeedForwardNetwork(d_model, d_ff)

#Forward pass
output = ffn.forward(x)

print("Input:", x)
print("\nAfter W1+b1 (before Relu):\n", np.round(x @ ffn.W1 + ffn.b1, 3))
print("\nAfter ReLU:\n", np.round(np.maximum(0, x @ ffn.W1 + ffn.b1), 3))
print("\nOutput after W2 + b2:\n", np.round(output, 3))
