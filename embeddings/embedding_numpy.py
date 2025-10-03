import numpy as np

class TransformerEmbedding:
  def __init__(self, vocab_size, d_model, max_len=500):
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.max_len = max_len

    #Initialize token embeddings randomly
    self.token_embedding = np.random.randn(vocab_size, d_model) * 0.01

    self.positional_encoding = self.generate_positional_encoding(max_len, d_model)


  def generate_positional_encoding(self, max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    k = np.arange(d_model)[np.newaxis, :]
    i = k//2

    angle_rads = pos/(np.power(10000, (2*i/(np.float32(d_model)))))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    positional_encoding = angle_rads[np.newaxis, ...]

    return positional_encoding


  def forward(self, input):
    batch_size, seq_len = input.shape


    token_embeddings = self.token_embedding[input]
    token_embeddings*=np.sqrt(self.d_model)
    positional_encoding = self.positional_encoding[:, :seq_len, :]

    return token_embeddings + positional_encoding