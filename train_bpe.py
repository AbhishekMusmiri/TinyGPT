import random
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tokenizer import BPETokenizer
from model import TinyGPT

# Keep this experiment separate from the original word-level model.
DATA_PATH = Path("training_data.txt")
TOKENIZER_PATH = Path("bpe_tokenizer.json")
CHECKPOINT_PATH = Path("tinygpt_bpe_checkpoint.pt")

SEED = 42
VOCAB_TARGET_SIZE = 300
BLOCK_SIZE = 16
BATCH_SIZE = 8
EPOCHS = 30
STEPS_PER_EPOCH = 100
LEARNING_RATE = 1e-3
PATIENCE = 5

random.seed(SEED)
torch.manual_seed(SEED)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

print("Loading training data...", flush=True)
text = DATA_PATH.read_text(encoding="utf-8")
sentences = [line.strip() for line in text.splitlines() if line.strip()]
if len(sentences) < 5:
    raise ValueError("training_data.txt must contain at least 5 non-empty lines.")

random.shuffle(sentences)
split_at = max(1, int(len(sentences) * 0.8))
train_sentences = sentences[:split_at]
val_sentences = sentences[split_at:]
if not val_sentences:
    val_sentences = train_sentences[-1:]
    train_sentences = train_sentences[:-1]

print(f"Loaded {len(sentences)} sentences.", flush=True)
print("Training BPE tokenizer...", flush=True)

tokenizer = BPETokenizer()
tokenizer.train("\n".join(train_sentences), vocab_size=VOCAB_TARGET_SIZE)

# Add explicit special tokens after the tokenizer's learned vocabulary.
for special in ("<pad>", "<unk>", "<eos>"):
    if special not in tokenizer.vocab:
        tokenizer.vocab[special] = len(tokenizer.vocab)

with TOKENIZER_PATH.open("w", encoding="utf-8") as f:
    json.dump({"vocab": tokenizer.vocab, "merges": tokenizer.merges}, f, ensure_ascii=False, indent=2)

vocab_size = len(tokenizer.vocab)
pad_id = tokenizer.vocab["<pad>"]
eos_id = tokenizer.vocab["<eos>"]
print(f"BPE vocabulary size: {vocab_size}", flush=True)

def make_examples(lines):
    examples = []
    for line in lines:
        ids = tokenizer.encode(line)
        ids = ids + [eos_id]
        # Predict each next token from up to BLOCK_SIZE preceding tokens.
        for target_index in range(1, len(ids)):
            context = ids[max(0, target_index - BLOCK_SIZE):target_index]
            target = ids[target_index]
            if context:
                examples.append((context, target))
    return examples

train_examples = make_examples(train_sentences)
val_examples = make_examples(val_sentences)
if not train_examples or not val_examples:
    raise ValueError("Could not create training/validation examples.")

print(f"Training examples: {len(train_examples)}", flush=True)
print(f"Validation examples: {len(val_examples)}", flush=True)

def make_batch(examples, sample=True):
    batch = random.sample(examples, min(BATCH_SIZE, len(examples))) if sample else examples
    xs, ys = [], []
    for context, target in batch:
        xs.append([pad_id] * (BLOCK_SIZE - len(context)) + context)
        ys.append(target)
    inputs = torch.tensor(xs, dtype=torch.long)
    return inputs, inputs.ne(pad_id), torch.tensor(ys, dtype=torch.long)

model = TinyGPT(
    vocab_size=vocab_size,
    d_model=32,
    max_seq_len=BLOCK_SIZE,
    num_layers=2,
    pad_token_id=pad_id,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

def calculate_loss(inputs, attention_mask, targets):
    logits = model(inputs, attention_mask=attention_mask)
    # All contexts are left-padded to BLOCK_SIZE, so the final column is the
    # final real context token in every example.
    return F.cross_entropy(logits[:, -1, :], targets)

@torch.no_grad()
def evaluate(examples):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for start in range(0, len(examples), BATCH_SIZE):
        batch = examples[start:start + BATCH_SIZE]
        inputs, attention_mask, targets = make_batch(batch, sample=False)
        total_loss += calculate_loss(inputs, attention_mask, targets).item() * len(batch)
        total_examples += len(batch)
    model.train()
    return total_loss / total_examples

best_val_loss = float("inf")
best_state = None
bad_epochs = 0

print("Starting TinyGPT training...", flush=True)
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for _ in range(STEPS_PER_EPOCH):
        inputs, attention_mask, targets = make_batch(train_examples)
        optimizer.zero_grad(set_to_none=True)
        loss = calculate_loss(inputs, attention_mask, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / STEPS_PER_EPOCH
    val_loss = evaluate(val_examples)
    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f}",
        flush=True,
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(
            {
                "model_state_dict": best_state,
                "vocab_size": vocab_size,
                "d_model": 32,
                "max_seq_len": BLOCK_SIZE,
                "num_layers": 2,
                "pad_token_id": pad_id,
                "best_val_loss": best_val_loss,
            },
            CHECKPOINT_PATH,
        )
        bad_epochs = 0
        print(f"  Saved best checkpoint to {CHECKPOINT_PATH}", flush=True)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("Early stopping: validation loss did not improve.", flush=True)
            break

print(f"Training finished. Best validation loss: {best_val_loss:.4f}", flush=True)
print(f"Tokenizer saved to: {TOKENIZER_PATH.resolve()}", flush=True)
print(f"Checkpoint saved to: {CHECKPOINT_PATH.resolve()}", flush=True)
