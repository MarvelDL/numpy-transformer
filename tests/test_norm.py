import numpy as np
from layers.normalization import AddNorm

d_model = 4
batch_size = 2
seq_len = 3

addnorm = AddNorm(d_model)

x = np.random.randn(batch_size, seq_len, d_model)
sublayer_out = np.random.randn(batch_size, seq_len, d_model)

print("Input:", x)
print("\nSublayer Output:\n", sublayer_out)
print("\nAfter Add&Norm:\n", addnorm.forward(x, sublayer_out))
