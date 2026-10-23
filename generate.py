
import torch
import torch.nn.functional as F
from model import TinyGPT


# Load the trained checkpoint
checkpoint = torch.load(
    "tinygpt_checkpoint.pt",
    map_location="cpu",
    weights_only=False
)

vocab = checkpoint["vocab"]
token_to_id = {token: i for i, token in enumerate(vocab)}
id_to_token = {i: token for i, token in enumerate(vocab)}

block_size = checkpoint["block_size"]

# Rebuild the model using the saved configuration
model = TinyGPT(
    vocab_size=len(vocab),
    d_model=checkpoint["d_model"],
    max_seq_len=block_size,
    num_layers=checkpoint["num_layers"],
    pad_token_id=checkpoint.get("pad_token_id"),
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def tokenize(text):
    """Use the same basic tokenizer as training."""
    import re
    return re.findall(r"\w+|[^\w\s]", text.lower())


def generate(
    prompt,
    max_new_tokens=20,
    temperature=0.8,
    top_k=5
):
    tokens = tokenize(prompt)

    if not tokens:
        raise ValueError("Please provide a non-empty prompt.")

    unknown = [token for token in tokens if token not in token_to_id]

    if unknown:
        raise ValueError(
            f"Unknown tokens: {unknown}. "
            "Use words from training_data.txt."
        )

    token_ids = torch.tensor(
        [[token_to_id[token] for token in tokens]],
        dtype=torch.long
    )

    eos_id = token_to_id.get("<eos>")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Keep only the most recent context
            context = token_ids[:, -block_size:]

            # Predict the next token
            logits = model(context)
            logits = logits[:, -1, :]

            pad_id = token_to_id.get("<pad>")
            if pad_id is not None:
                logits[:, pad_id] = float("-inf")

            # Avoid invalid temperature values
            if temperature <= 0:
                raise ValueError("Temperature must be greater than zero.")

            logits = logits / temperature

            # Top-k filtering
            k = max(1, min(top_k, logits.size(-1)))
            values, indices = torch.topk(logits, k=k, dim=-1)

            filtered_logits = torch.full_like(
                logits,
                float("-inf")
            )
            filtered_logits.scatter_(1, indices, values)

            # Convert scores into probabilities
            probabilities = F.softmax(filtered_logits, dim=-1)

            # Sample the next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Stop when the model predicts end-of-sentence
            if eos_id is not None and next_token.item() == eos_id:
                break

            # Add the generated token to the sequence
            token_ids = torch.cat(
                [token_ids, next_token],
                dim=1
            )

    generated_tokens = [
        id_to_token[token_id]
        for token_id in token_ids[0].tolist()
    ]

    # Hide special tokens in the displayed output
    generated_tokens = [
        token for token in generated_tokens
        if token not in ("<eos>", "<pad>")
    ]

    return " ".join(generated_tokens)


if __name__ == "__main__":
    torch.manual_seed(42)

    print("Generated:", generate(
        "learning",
        max_new_tokens=15,
        temperature=0.8,
        top_k=5
    ))

    print("Generated:", generate(
        "Artificial",
        max_new_tokens=15,
        temperature=0.8,
        top_k=5
    ))
