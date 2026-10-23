
import torch
import torch.nn as nn

from transformer_block import TransformerBlock


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=32,
        max_seq_len=64,
        num_layers=2,
        pad_token_id=None,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # ModuleList keeps the existing ``blocks.<n>`` checkpoint key layout
        # while allowing an attention mask to reach every block.
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, attention_mask=None):
        # token_ids shape: (batch, sequence)
        batch_size, seq_len = token_ids.shape

        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError("Sequence exceeds max_seq_len")

        if attention_mask is None:
            positions = torch.arange(seq_len, device=token_ids.device)
        else:
            if attention_mask.shape != token_ids.shape:
                raise ValueError("attention_mask must match token_ids")

            # With left padding, assign real tokens positions 0, 1, ... so
            # their representations match unpadded generation contexts.
            positions = attention_mask.to(dtype=torch.long).cumsum(dim=1) - 1
            positions = positions.clamp_min(0).to(token_ids.device)

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(positions)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.final_norm(x)

        logits = self.output_layer(x)
        return logits


if __name__ == "__main__":
    torch.manual_seed(42)

    vocab_size = 100
    model = TinyGPT(vocab_size=vocab_size)

    # Batch of 2 sequences, 5 tokens per sequence
    token_ids = torch.randint(0, vocab_size, (2, 5))

    logits = model(token_ids)

    print("Input shape:", token_ids.shape)
    print("Logits shape:", logits.shape)
    print("Model parameters:", sum(p.numel() for p in model.parameters()))
