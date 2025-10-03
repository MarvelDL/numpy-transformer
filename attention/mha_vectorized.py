import numpy as np
from attention.scaled_dot_product import scaled_dot_product_attention

class MultiHeadAttention:

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0 #d_model must be divisible by num_heads

        self.d_model = d_model
        self.num_heads =  num_heads
        self.d_k = d_model//num_heads

        self.W_Q = np.random.randn(d_model, d_model)*0.01
        self.W_K = np.random.randn(d_model, d_model)*0.01
        self.W_V = np.random.randn(d_model, d_model)*0.01
        self.W_O = np.random.randn(d_model, d_model)*0.01

    
    def split_heads(self, x, batch_size):

        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)

        return x.transpose(0, 2, 1, 3) #swap num_heads and seq_len
    
    
    def combine_heads(self, x, batch_size): #Inverse of split_heads

        x = x.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)

        return x
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        #Linear projections
        Q_proj = Q @ self.W_Q #matmul
        K_proj = K @ self.W_K
        V_proj = V @ self.W_V

        #Split into heads
        Q_heads = self.split_heads(Q_proj, batch_size)
        K_heads = self.split_heads(K_proj, batch_size)
        V_heads = self.split_heads(V_proj, batch_size)

        #Vectorization instead of for-loop
        B_H = batch_size * self.num_heads
        Q_ = Q_heads.reshape(B_H, -1, self.d_k) #(batch_size*num_heads, seq_len, d_k)
        K_ = K_heads.reshape(B_H, -1, self.d_k)
        V_ = V_heads.reshape(B_H, -1, self.d_k)

        out, attn = scaled_dot_product_attention(Q_, K_, V_, mask)
        #out = (B_H, seq_len, d_k), attn = (B_H, seq_len, seq_len)

        #Reshaping back to (batch_size, num_heads, seq_len, d_k)
        out = out.reshape(batch_size, self.num_heads, -1, self.d_k)
        attn = attn.reshape(batch_size, self.num_heads, -1, K_.shape[1])

        #Combine heads + final projection
        concat = self.combine_heads(out, batch_size)
        output = concat @ self.W_O

        return output, attn



        