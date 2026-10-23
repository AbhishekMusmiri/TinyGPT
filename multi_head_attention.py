
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads"
            )

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None):
        """Apply causal self-attention.

        ``attention_mask`` is an optional boolean tensor of shape
        ``(batch, sequence)`` where ``True`` identifies a real token. It
        prevents left-padding tokens from contributing keys or values.
        """
        batch_size, seq_len, d_model = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split features across attention heads
        Q = Q.view(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size, seq_len,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Each head calculates its own attention scores
        scores = (Q @ K.transpose(-2, -1)) / (
            self.head_dim ** 0.5
        )

        future_mask = torch.triu(
            torch.ones(
                seq_len, seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(future_mask, float("-inf"))

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError("attention_mask must have shape (batch, sequence)")

            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            key_padding_mask = ~attention_mask[:, None, None, :]
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        # Padded query positions have no valid keys. Keep them finite even
        # though their outputs are not used for next-token prediction.
        weights = torch.nan_to_num(weights, nan=0.0)

        if attention_mask is not None:
            weights = weights * attention_mask[:, None, :, None]

        output = weights @ V

        # Join all heads back together
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, d_model)

        return self.output_projection(output)


if __name__ == "__main__":
    torch.manual_seed(42)

    x = torch.randn(2, 4, 8)

    attention = MultiHeadCausalSelfAttention(
        d_model=8,
        num_heads=2
    )

    output = attention(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Head dimension:", attention.head_dim)
