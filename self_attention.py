
import math
import torch
import torch.nn as nn

torch.manual_seed(42)

# Four tokens, each represented by four features
x = torch.randn(4, 4)

# Three learned transformations
query_layer = nn.Linear(4, 4, bias=False)
key_layer = nn.Linear(4, 4, bias=False)
value_layer = nn.Linear(4, 4, bias=False)

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

# Compare each query with every key
scores = Q @ K.T

# Scale the scores
scores = scores / math.sqrt(K.shape[-1])

# Convert scores into attention weights
weights = torch.softmax(scores, dim=-1)

# Combine value vectors using those weights
output = weights @ V

print("Input shape:", x.shape)
print("Attention scores shape:", scores.shape)
print("Attention weights:")
print(weights)
print("Output shape:", output.shape)
print("Output:")
print(output)
