import numpy as np
from models.decoder import Decoder
from models.encoder import Encoder

class Transformer:
    def __init__(self, d_model, num_heads, d_ff, num_layers, src_vocab_size, tgt_vocab_size, max_len = 500):

        self.encoder = Encoder(
            d_model = d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            vocab_size=src_vocab_size,
            max_len=max_len,
            num_layers=num_layers
        )

        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            tgt_vocab_size=tgt_vocab_size,
            max_len=max_len
        )

        self.final_layer = np.random.randn(d_model, tgt_vocab_size)*0.01

    def forward(self, src_ids, tgt_ids):

        enc_output, enc_attn =  self.encoder.forward(src_ids)

        dec_output, dec_attn = self.decoder.forward(tgt_ids, enc_output, enc_input_ids=src_ids)

        logits = dec_output@self.final_layer

        return logits, {"enc_attn": enc_attn, "dec_attn": dec_attn}
