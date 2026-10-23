
import re
import random
import torch
import torch.nn.functional as F

from model import TinyGPT

# -------------------------
# 1. Configuration
# -------------------------
SEED = 42
BLOCK_SIZE = 16
BATCH_SIZE = 8
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 12
MIN_IMPROVEMENT = 0.01

torch.manual_seed(SEED)
random.seed(SEED)

# -------------------------
# 2. Load sentence-based data
# -------------------------
with open("training_data.txt", "r", encoding="utf-8") as file:
    sentences = [
        line.strip().lower()
        for line in file
        if line.strip()
    ]

if len(sentences) < 5:
    raise ValueError("Add at least five sentences to training_data.txt.")

random.shuffle(sentences)

# Keep at least one sentence for validation
split_index = max(1, int(len(sentences) * 0.8))
split_index = min(split_index, len(sentences) - 1)

train_sentences = sentences[:split_index]
val_sentences = sentences[split_index:]

# -------------------------
# 3. Tokenizer and vocabulary
# -------------------------
def tokenize(sentence):
    return re.findall(r"\w+|[^\w\s]", sentence.lower())


# Build vocabulary from training data only
vocab = sorted({
    token
    for sentence in train_sentences
    for token in tokenize(sentence)
})

for special_token in ["<pad>", "<eos>", "<unk>"]:
    if special_token not in vocab:
        vocab.append(special_token)

token_to_id = {
    token: index for index, token in enumerate(vocab)
}

pad_id = token_to_id["<pad>"]
eos_id = token_to_id["<eos>"]
unk_id = token_to_id["<unk>"]

# -------------------------
# 4. Build next-token examples
# -------------------------
def encode_sentence(sentence):
    ids = [
        token_to_id.get(token, unk_id)
        for token in tokenize(sentence)
    ]
    ids.append(eos_id)
    return ids


def make_examples(sentence_list):
    examples = []

    for sentence in sentence_list:
        ids = encode_sentence(sentence)

        # Predict one next token from preceding context
        for position in range(1, len(ids)):
            start = max(0, position - BLOCK_SIZE)

            context = ids[start:position]
            target = ids[position]

            examples.append((context, target))

    return examples


train_examples = make_examples(train_sentences)
val_examples = make_examples(val_sentences)

if not train_examples or not val_examples:
    raise ValueError("Could not create training or validation examples.")

# -------------------------
# 5. Prepare mini-batches
# -------------------------
def make_batch(examples, sample=True):
    batch = random.sample(
        examples,
        min(BATCH_SIZE, len(examples))
    ) if sample else examples

    inputs = []
    targets = []

    for context, target in batch:
        padding = BLOCK_SIZE - len(context)

        inputs.append([pad_id] * padding + context)
        targets.append(target)

    inputs = torch.tensor(inputs, dtype=torch.long)
    return inputs, inputs.ne(pad_id), torch.tensor(targets, dtype=torch.long)


# -------------------------
# 6. Create the model
# -------------------------
model = TinyGPT(
    vocab_size=len(vocab),
    d_model=32,
    max_seq_len=BLOCK_SIZE,
    num_layers=2,
    pad_token_id=pad_id,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)

# Ignore left-padding tokens when predicting.
# Each target is the next token after the last real input token.
def calculate_loss(inputs, attention_mask, targets):
    logits = model(inputs, attention_mask=attention_mask)
    return F.cross_entropy(logits[:, -1, :], targets)


# -------------------------
# 7. Validation
# -------------------------
def evaluate(examples):
    model.eval()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for start in range(0, len(examples), BATCH_SIZE):
            batch = examples[start:start + BATCH_SIZE]
            inputs, attention_mask, targets = make_batch(batch, sample=False)

            loss = calculate_loss(inputs, attention_mask, targets)

            total_loss += loss.item() * len(batch)
            total_examples += len(batch)

    model.train()
    return total_loss / total_examples


# -------------------------
# 8. Training with early stopping
# -------------------------
best_val_loss = float("inf")
epochs_without_improvement = 0

print("Training sentences:", len(train_sentences))
print("Validation sentences:", len(val_sentences))
print("Training examples:", len(train_examples))
print("Validation examples:", len(val_examples))
print("Vocabulary size:", len(vocab))

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    random.shuffle(train_examples)

    total_train_loss = 0.0
    total_train_examples = 0

    for start in range(0, len(train_examples), BATCH_SIZE):
        batch = train_examples[start:start + BATCH_SIZE]
        inputs, attention_mask, targets = make_batch(batch, sample=False)

        loss = calculate_loss(inputs, attention_mask, targets)

        optimizer.zero_grad()
        loss.backward()

        # Helps prevent excessively large gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_train_loss += loss.item() * len(batch)
        total_train_examples += len(batch)

    train_loss = total_train_loss / total_train_examples
    val_loss = evaluate(val_examples)

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {val_loss:.4f}"
        )

    # Save the best model, not automatically the last model
    if val_loss < best_val_loss - MIN_IMPROVEMENT:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "block_size": BLOCK_SIZE,
                "d_model": 32,
                "num_layers": 2,
                "pad_token_id": pad_id,
                "best_val_loss": best_val_loss,
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "unk_token": "<unk>"
            },
            "tinygpt_checkpoint.pt"
        )
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break

print("Training complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("Best checkpoint saved: tinygpt_checkpoint.pt")
