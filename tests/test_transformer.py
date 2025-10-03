import numpy as np
from models.transformer import Transformer

config = {
    "d_model": 32,
    "num_heads": 4,
    "d_ff": 64,
    "num_layers": 2,
    "src_vocab_size": 50,
    "tgt_vocab_size": 60,
    "max_len": 20
}

transformer = Transformer(**config)

src = np.array([[1, 5, 6, 2, 0, 0],
                [4, 3, 9, 10, 11, 2]])
tgt = np.array([[1, 7, 8, 2, 0],
                [1, 4, 5, 6, 2]])

logits, attns = transformer.forward(src, tgt)

print("Source IDs:\n", src)
print("Target IDs:\n", tgt)
print("Logits Shape:\n", logits.shape)
print("\nEncoder attention layers:", len(attns["enc_attn"]))
print("Decoder self-attn layers:", len(attns["dec_attn"]["dec_self"]))
print("Decoder enc-dec attn layers:", len(attns["dec_attn"]["enc_dec"]))