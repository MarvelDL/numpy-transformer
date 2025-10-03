import numpy as np
from attention.mha_vectorized import MultiHeadAttention

def test_mha():
    batch_size, seq_len, d_model, num_heads = 1, 4, 8, 2
    mha = MultiHeadAttention(d_model, num_heads)

    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    output, attn_weights = mha.forward(Q, K, V)

    print("Output Shape:", output.shape)
    print("Output:", output)
    print("Number of Attention weights:", attn_weights)
    print("Attention Weights:", attn_weights)

if __name__== "__main__":
    test_mha()
