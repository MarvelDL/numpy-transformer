import numpy as np
from attention.mha_vectorized import MultiHeadAttention
from feed_forward.ffn import FeedForwardNetwork
from layers.normalization import AddNorm
from embeddings.embedding_numpy import TransformerEmbedding
from utils.masks import create_padding_mask

class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.d_model= d_model
        self.num_heads = num_heads

        self.mha =  MultiHeadAttention(d_model, num_heads)
        self.addnorm1 = AddNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.addnorm2 = AddNorm(d_model)

    def prepare_mask_for_mha(self, mask, batch_size):
        if mask is None:
            return None
        
        mask_squeezed = np.squeeze(mask, axis=1)

        if mask_squeezed.shape[0] == 1 and batch_size > 1:
            mask_squeezed = np.repeat(mask_squeezed, repeats=batch_size, axis=0)

        mask_for_mha = np.repeat(mask_squeezed, repeats=self.num_heads, axis=0)
        return mask_for_mha
    
    
    def forward(self, x, padding_mask=None):

        B, src_T, _ = x.shape
        
        mask_for_mha = self.prepare_mask_for_mha(padding_mask, B)
        
        #MultiHead Attention
        attn_outputs, attn_weight = self.mha.forward(x, x, x, mask_for_mha)

        if attn_weight is not None:
            attn_weight = attn_weight.reshape(B, self.num_heads, src_T, src_T)


        #Add and Norm
        out1 = self.addnorm1.forward(x, attn_outputs)

        #Feed-Forward NN
        output = self.ffn.forward(out1)

        #Second Add and Norm
        out2 = self.addnorm2.forward(out1, output)

        return out2, attn_weight


class Encoder:

    def __init__(self, d_model, num_heads, d_ff, vocab_size,
                 max_len, num_layers):
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.num_heads =  num_heads

        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len)

        self.layers =  [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]


    def forward(self, x, mask=None):

        pad = create_padding_mask(x)
        x = self.embedding.forward(x)

        attn_weights_all = []

        for layer in self.layers:
            x, attn_weights = layer.forward(x, pad)
            attn_weights_all.append(attn_weights)

        return x, attn_weights_all

    

