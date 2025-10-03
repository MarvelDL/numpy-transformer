import numpy as np
from embeddings.embedding_numpy import TransformerEmbedding
from attention.mha_vectorized import MultiHeadAttention
from feed_forward.ffn import FeedForwardNetwork
from layers.normalization import AddNorm
from utils.masks import combine_masks, create_look_ahead_mask, create_padding_mask


class DecoderBlock:

    def __init__(self, d_model, num_heads, d_ff):

        self.d_model = d_model
        self.num_heads = num_heads

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)

        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)
        self.addnorm3 = AddNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model, d_ff)

    
    def prepare_masks_for_mha(self, mask, batch_size):

        if mask is None:
            return None
        
        mask_squeezed = np.squeeze(mask, axis=1)

        if mask_squeezed.shape[0] == 1 and batch_size > 1:
            mask_squeezed = np.repeat(mask_squeezed, repeats=batch_size, axis=0)
        
        mask_for_mha = np.repeat(mask_squeezed, repeats=self.num_heads, axis=0)
        return mask_for_mha

    def forward(self, x, enc_output, look_ahead_mask=None, tgt_padding_mask=None, enc_padding_mask=None):

        B = x.shape[0]
        tgt_T = x.shape[1]
        src_T = enc_output.shape[1]

        combined_mask = combine_masks(look_ahead_mask, tgt_padding_mask)

        mask_for_self = self.prepare_masks_for_mha(combined_mask, B)

        #Masked Self-Attention
        mha1_out, attn1= self.self_attn.forward(x, x, x, mask_for_self)

        #If attn1 shape = (B*H, Tt, Tt), reshape to (B, H, Tt, Tt)
        if attn1 is not None:
            attn1 = attn1.reshape(B, self.num_heads, tgt_T, tgt_T)

        out1 = self.addnorm1.forward(x, mha1_out)

        #Encoder-Decoder Attention
        if enc_padding_mask is not None:
            enc_pad_expanded = np.repeat(enc_padding_mask, repeats=tgt_T, axis=2)
        else:
            enc_pad_expanded = None
        
        mask_for_encdec = self.prepare_masks_for_mha(enc_pad_expanded, B)

        mha2_out, attn2 = self.enc_dec_attn.forward(out1, enc_output, enc_output, mask_for_encdec)

        if attn2 is not None:
            attn2 = attn2.reshape(B, self.num_heads, tgt_T, src_T)
        
        out2 = self.addnorm2.forward(out1, mha2_out)

        #Feed Forward
        ffn_out= self.ffn.forward(out2)
        out3 = self.addnorm3.forward(out2, ffn_out)

        return out3, attn1, attn2
    

class Decoder:

    def __init__(self, d_model, num_heads, d_ff, num_layers, tgt_vocab_size, max_len=500):
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers= num_layers
        self.vocab_size = tgt_vocab_size
        self.max_len = max_len

        self.embedding = TransformerEmbedding(self.vocab_size, self.d_model, max_len=self.max_len)

        self.layers = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, tar_ids, enc_output, enc_input_ids=None):

        B, tgt_T = tar_ids.shape

        #Create Look-Ahead-Mask
        look_ahead = create_look_ahead_mask(tgt_T)

        #Padding Mask (for decoder self-attention)
        tgt_pad = create_padding_mask(tar_ids)

        #Encoder padding mask (for enc-dec attention)
        enc_pad = None
        if enc_input_ids is not None:
            enc_pad = create_padding_mask(enc_input_ids)
        
        #Embed targets
        x = self.embedding.forward(tar_ids)

        dec_self_attns = []
        enc_dec_attns = []

        for layer in self.layers:
            x, attn1, attn2 = layer.forward(x, enc_output, look_ahead, tgt_pad, enc_pad)
            
            dec_self_attns.append(attn1)
            enc_dec_attns.append(attn2)

        return x, {"dec_self": dec_self_attns, "enc_dec": enc_dec_attns}
    




