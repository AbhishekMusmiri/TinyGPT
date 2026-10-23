
import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch, sequence, features)
        batch_size, seq_len, d_model = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_model)

        # Block attention to future positions
        future_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )

        scores = scores.masked_fill(future_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        output = weights @ V

        return self.output_projection(output)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Batch of 2 sequences, 4 tokens per sequence, 8 features per token
    x = torch.randn(2, 4, 8)

    attention = CausalSelfAttention(d_model=8)
    output = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
