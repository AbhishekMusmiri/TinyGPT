
import torch
import torch.nn as nn

# Token IDs produced by our tokenizer
token_ids = torch.tensor([3, 5, 2, 0])

# Our vocabulary contains 6 unique tokens
vocab_size = 6

# Represent each token using 4 numbers
embedding_dim = 4

# Create the embedding table
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Look up a vector for every token ID
embeddings = embedding_layer(token_ids)

print("Token IDs:", token_ids)
print("Embedding shape:", embeddings.shape)
print("Embeddings:")
print(embeddings)
