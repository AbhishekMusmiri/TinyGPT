
import torch
import torch.nn as nn
from multi_head_attention import MultiHeadCausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadCausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.norm1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    block = TransformerBlock(d_model=32, num_heads=4)
    x = torch.randn(2, 4, 32)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
