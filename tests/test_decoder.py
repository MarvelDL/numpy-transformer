import numpy as np
from models.decoder import Decoder


np.random.seed(0)

batch_size = 2
src_len = 6
tgt_len = 5
d_model = 8
num_heads = 2
d_ff = 16
num_layers = 2
vocab_size = 50
max_len = 32

# dummy token ids (PAD=0) 
encoder_input_ids = np.array([
    [10, 11, 12, 0, 0, 0],
    [7, 8, 9, 10, 0, 0]
])

decoder_input_ids = np.array([
    [1, 2, 3, 0, 0],
    [4, 5, 6, 7, 0]
])

# Fake encoder output embeddings (pretend encoder produced these)
enc_output = np.random.randn(batch_size, src_len, d_model).astype(np.float32)

# decoder
decoder = Decoder(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                  num_layers=num_layers, tgt_vocab_size=vocab_size, max_len=max_len)

# Forward pass
out, all_attn = decoder.forward(decoder_input_ids, enc_output, enc_input_ids=encoder_input_ids)

print("Decoder output shape:", out.shape)
print("dec_self attn layers:", len(all_attn["dec_self"]))
print("enc_dec attn layers:", len(all_attn["enc_dec"]))
if all_attn["dec_self"][0] is not None:
    print("dec_self attn shape (layer0):", all_attn["dec_self"][0].shape)  
if all_attn["enc_dec"][1] is not None:
    print("enc_dec attn shape (layer1):", all_attn["enc_dec"][1].shape)