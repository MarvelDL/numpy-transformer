# Transformer from Scratch (NumPy Implementation)

This project is a **from-scratch implementation of the Transformer architecture** (Vaswani et al., 2017) using **NumPy only**.  
The goal is educational: to understand every moving part of the Transformer without relying on deep learning frameworks like PyTorch or TensorFlow.  

---

## Features Implemented

- **Embeddings**: Token + positional embeddings.  
- **Encoder**:
  - Multi-Head Attention (vectorized NumPy implementation).  
  - Feed-Forward Network.  
  - Add & Norm layers.  
  - Supports **padding masks**.  
- **Decoder**:
  - Masked Multi-Head Self-Attention.  
  - Encoder–Decoder Attention.  
  - Feed-Forward Network.  
  - Add & Norm layers.  
  - Supports **look-ahead masks** (to prevent cheating) and **padding masks**.  
- **Full Transformer**:
  - Encoder + Decoder stack.  
  - Output projection to vocabulary logits.  

---

## Project Structure

```
numpy-transformer/
│
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   └── transformer.py
│
├── attention/
│   ├── mha_vectorized.py
│   └── scaled_dot_product.py
│
├── layers/
│   └── normalization.py
│   
│
├── embeddings/
│   └── embedding_numpy.py
│
├── utils/
│   ├── masks.py
│   └── utils_numpy.py
│
├── tests/
│   ├── test_attention.py
│   ├── test_mha.py
│   ├── test_ffn.py
│   ├── test_encoder.py
│   ├── test_norm.py
│   ├── test_decoder.py
│   └── test_transformer.py
│
└── README.md
```

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/MarvelDL/numpy-transformer.git
   cd numpy-transformer
   ```

2. Run tests for different components:

   - **Attention:**
     ```bash
     python -m tests.test_attention
     ```

   - **Multi-Head Attention:**
     ```bash
     python -m tests.test_mha
     ```

   - **Feed Forward Network:**
     ```bash
     python -m tests.test_ffn
     ```

   - **Normalization:**
     ```bash
     python -m tests.test_norm
     ```

   - **Encoder:**
     ```bash
     python -m tests.test_encoder
     ```

   - **Decoder:**
     ```bash
     python -m tests.test_decoder
     ```

   - **Full Transformer:**
     ```bash
     python -m tests.test_transformer
     ```

---

## Notes

- **No training yet**: This project only covers the **forward pass** of the Transformer (NumPy makes training inefficient).  
- **Masks**:  
  - **Look-ahead mask** ensures autoregressive decoding by preventing access to future tokens.  
  - **Padding mask** ensures padded tokens don’t affect attention.  
- Next steps could include implementing this in **PyTorch**, **TensorFlow**, or **JAX** for training.  

---

## Reference
- Vaswani et al., *Attention Is All You Need*, 2017.  
