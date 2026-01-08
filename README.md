````markdown
# Transformer-From-Scratch

A clean, well-documented **PyTorch implementation of the Transformer architecture**
from the paper **“Attention Is All You Need” (Vaswani et al., 2017)**.

This project is designed for **learning, clarity, and extensibility**, implementing
the Transformer completely from scratch without relying on high-level abstractions.

---

## ✨ Features

- Complete Transformer Architecture (Encoder + Decoder)
- Multi-Head Attention with configurable heads
- Sinusoidal Positional Encoding
- Pre-Layer Normalization for training stability
- Modular and extensible design
- Full type hints for readability

---

## 🧠 Architecture Components

- `InputEmbedding` – Token embeddings with scaling
- `PositionalEncoding` – Sinusoidal positional encodings
- `MultiHeadAttention` – Scaled dot-product attention
- `FeedForward` – Position-wise feed-forward network
- `LayerNormalization` – Custom LayerNorm
- `ResidualConnection` – Residual connections with dropout
- `EncoderBlock` & `DecoderBlock`
- `Encoder` & `Decoder`
- `Transformer` – Complete model

---

## 📦 Installation

```bash
git clone https://github.com/Amit95688/Transformer-From-Scratch.git
cd Transformer-From-Scratch
pip install torch
````

---

## ⚙️ Requirements

* Python 3.7+
* PyTorch 1.9+

---

## 🚀 Usage

```python
import torch
from model import build_transformer

src_vocab_size = 10000
tgt_vocab_size = 10000
src_seq_len = 100
tgt_seq_len = 100

model = build_transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    src_seq_len=src_seq_len,
    tgt_seq_len=tgt_seq_len,
    d_model=512,
    d_ff=2048,
    num_heads=8,
    num_layers=6,
    dropout=0.1
)

src = torch.randint(0, src_vocab_size, (32, 20))
tgt = torch.randint(0, tgt_vocab_size, (32, 20))

encoder_output = model.encode(src)
decoder_output = model.decode(tgt, encoder_output)
output = model.project(decoder_output)

print(output.shape)  # [32, 20, tgt_vocab_size]
```

---

## 🧩 Individual Components

```python
import torch
from model import MultiHeadAttention, FeedForward, EncoderBlock

d_model = 512

self_attn = MultiHeadAttention(d_model, num_heads=8, dropout=0.1)
ff = FeedForward(d_model, d_ff=2048, dropout=0.1)

encoder_block = EncoderBlock(d_model, self_attn, ff, dropout=0.1)

x = torch.randn(32, 20, d_model)
output = encoder_block(x)
```

---

## 🎭 Masks

```python
def create_padding_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0
```

---

## 🔧 Model Parameters

| Parameter      | Type  | Default | Description                |
| -------------- | ----- | ------- | -------------------------- |
| src_vocab_size | int   | —       | Source vocabulary size     |
| tgt_vocab_size | int   | —       | Target vocabulary size     |
| src_seq_len    | int   | —       | Max source sequence length |
| tgt_seq_len    | int   | —       | Max target sequence length |
| d_model        | int   | 512     | Model dimension            |
| d_ff           | int   | 2048    | Feed-forward dimension     |
| num_heads      | int   | 8       | Attention heads            |
| num_layers     | int   | 6       | Encoder/Decoder layers     |
| dropout        | float | 0.1     | Dropout probability        |

---

## 📐 Architecture Math

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

---

## 📁 Project Structure

.
├── model.py
└── README.md

---

## 📜 License

MIT License

---

## 📚 Citation

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish et al.},
  journal={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```

---

## 📬 Contact

[https://github.com/Amit95688/Transformer-From-Scratch](https://github.com/Amit95688/Transformer-From-Scratch)

```
```
