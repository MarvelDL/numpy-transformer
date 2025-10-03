import numpy as np
from attention.scaled_dot_product import scaled_dot_product_attention

def test_attention():
    batch_size, seq_len, d_k, d_v = 2, 4, 8, 8

    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)

    output, attn_weights = scaled_dot_product_attention(Q, K, V)
    print("Output Shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape)

if __name__ == "__main__":
    test_attention()
