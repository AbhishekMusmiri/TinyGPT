# TinyGPT

TinyGPT is a small, educational decoder-only Transformer written in PyTorch.
It is designed to show the essential parts of GPT-style next-token prediction
without hiding the model behind a large framework.

The maintained training path uses a simple Byte Pair Encoding (BPE) tokenizer,
causal multi-head self-attention, learned positional embeddings, and a small
two-layer Transformer model.

## What is included

- `model.py` — token/position embeddings and the TinyGPT model
- `transformer_block.py` — pre-normalized attention and feed-forward block
- `multi_head_attention.py` — masked causal multi-head self-attention
- `tokenizer.py` — minimal BPE tokenizer implementation
- `train_bpe.py` — train the BPE tokenizer and TinyGPT checkpoint
- `generate_bpe.py` — generate text from the trained BPE checkpoint
- `training_data.txt` — small demonstration dataset

The repository also contains earlier word-level and attention-only examples for
learning purposes.

## Requirements

- Python 3.10 or newer
- PyTorch

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch
```

## Train

The trained model and tokenizer are generated locally and intentionally are not
committed to Git. Train them with:

```powershell
.\.venv\Scripts\python.exe train_bpe.py
```

This creates:

- `bpe_tokenizer.json`
- `tinygpt_bpe_checkpoint.pt`

## Generate text

After training, run:

```powershell
.\.venv\Scripts\python.exe generate_bpe.py
```

Edit `PROMPTS`, `TEMPERATURE`, `TOP_K`, and `MAX_NEW_TOKENS` in
`generate_bpe_fixed.py` to experiment with generation.

## Model details

The default BPE model is deliberately small:

| Setting | Value |
| --- | ---: |
| Embedding dimension | 32 |
| Transformer layers | 2 |
| Attention heads | 4 |
| Context length | 16 tokens |
| Target BPE vocabulary | 300 tokens |

Training applies a causal mask so a token cannot attend to future tokens. It
also masks padding tokens and assigns left-padded sequences consistent position
IDs, so batching matches generation behavior.

## Limitations

`training_data.txt` contains only 500 short, synthetic sentences. The model is
therefore expected to produce fragments rather than fluent text. To improve
generation, use a larger coherent dataset, then increase the context length,
embedding dimension, and number of layers.

## License

No license has been selected yet.
