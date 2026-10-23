import math
import torch
import torch.nn as nn

torch.manual_seed(42)

x = torch.randn(4, 4)

query_layer = nn.Linear(4, 4, bias=False)
key_layer = nn.Linear(4, 4, bias=False)
value_layer = nn.Linear(4, 4, bias=False)

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

scores = (Q @ K.T) / math.sqrt(K.shape[-1])

# True means that this position is in the future
future_mask = torch.triu(
    torch.ones(4, 4, dtype=torch.bool),
    diagonal=1
)

# Prevent attention to future positions
scores = scores.masked_fill(future_mask, float("-inf"))

weights = torch.softmax(scores, dim=-1)
output = weights @ V

print("Causal mask:")
print(future_mask.int())

print("\nAttention weights:")
print(weights)

print("\nOutput shape:", output.shape)
