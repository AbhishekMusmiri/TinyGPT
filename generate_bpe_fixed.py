import json
import os
import random

import torch

from model import TinyGPT
from tokenizer import BPETokenizer

CHECKPOINT_PATH = "tinygpt_bpe_checkpoint.pt"
TOKENIZER_PATH = "bpe_tokenizer.json"

PROMPTS = [
    "artificial intelligence",
    "transformers",
    "python helps developers",
]

MAX_NEW_TOKENS = 30
TEMPERATURE = 0.8
TOP_K = 20
SEED = 42


def load_tokenizer(path):
    tokenizer = BPETokenizer()
    with open(path, "r", encoding="utf-8") as file:
        saved = json.load(file)

    tokenizer.vocab = saved["vocab"]
    tokenizer.merges = saved["merges"]
    return tokenizer


def generate(model, tokenizer, prompt, max_seq_len, max_new_tokens=30,
             temperature=0.8, top_k=20):
    prompt_ids = tokenizer.encode(prompt)
    token_ids = list(prompt_ids)

    if not token_ids:
        token_ids = [tokenizer.vocab.get("<unk>", 0)]

    eos_id = tokenizer.vocab.get("<eos>")
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = token_ids[-max_seq_len:]
            inputs = torch.tensor([context], dtype=torch.long)
            logits = model(inputs)[0, -1, :] / max(temperature, 1e-5)

            # Padding was never a target and is not valid generated text.
            pad_id = tokenizer.vocab.get("<pad>")
            if pad_id is not None:
                logits[pad_id] = float("-inf")

            if top_k is not None and 0 < top_k < logits.numel():
                values, _ = torch.topk(logits, top_k)
                cutoff = values[-1]
                logits[logits < cutoff] = float("-inf")

            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1).item()
            token_ids.append(next_id)

            if eos_id is not None and next_id == eos_id:
                break

    generated_ids = token_ids[len(prompt_ids):]
    return tokenizer.decode(generated_ids).strip()


def main():
    torch.manual_seed(SEED)
    for path in (CHECKPOINT_PATH, TOKENIZER_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path}. Make sure it is in the TinyGPT project folder."
            )

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        d_model=checkpoint["d_model"],
        max_seq_len=checkpoint["max_seq_len"],
        num_layers=checkpoint["num_layers"],
        pad_token_id=checkpoint.get("pad_token_id"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded BPE TinyGPT checkpoint.")
    print(f"Vocabulary size: {checkpoint['vocab_size']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print()

    for prompt in PROMPTS:
        completion = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_seq_len=checkpoint["max_seq_len"],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
        )
        print(f"Prompt:    {prompt}")
        print(f"Generated: {completion or '[no visible text generated]'}")
        print()


if __name__ == "__main__":
    main()
