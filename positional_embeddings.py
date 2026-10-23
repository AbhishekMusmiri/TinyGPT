
import torch
import torch.nn as nn

# Example: 4 tokens, each represented by 4 numbers
token_ids = torch.tensor([3, 5, 2, 0])

vocab_size = 6
embedding_dim = 4
max_sequence_length = 16

# Learns a vector for each token ID
token_embedding = nn.Embedding(vocab_size, embedding_dim)

# Learns a vector for each position
position_embedding = nn.Embedding(
    max_sequence_length, embedding_dim
)

# Positions: 0, 1, 2, 3
positions = torch.arange(len(token_ids))

# Look up both sets of vectors
token_vectors = token_embedding(token_ids)
position_vectors = position_embedding(positions)

# Combine token identity and position
x = token_vectors + position_vectors

print("Token vectors:", token_vectors.shape)
print("Position vectors:", position_vectors.shape)
print("Combined shape:", x.shape)
print(x)
