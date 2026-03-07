"""Text autoencoder model definitions."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextAutoencoder(nn.Module):
    """A character-level sequence autoencoder built with GRUs."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        latent_dim: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than 0")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be greater than 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def encode(self, input_ids: Tensor, lengths: Tensor) -> Tensor:
        """Encode padded input sequences into latent vectors."""
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        return self.to_latent(hidden[-1])

    def decode(self, latent: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Decode latent vectors into token logits with teacher forcing inputs."""
        decoder_hidden = self.from_latent(latent).unsqueeze(0)
        embedded = self.embedding(decoder_input_ids)
        outputs, _ = self.decoder(embedded, decoder_hidden)
        return self.output_layer(outputs)

    def forward(self, input_ids: Tensor, lengths: Tensor, decoder_input_ids: Tensor) -> Tensor:
        """Run the autoencoder forward pass and return token logits."""
        latent = self.encode(input_ids, lengths)
        return self.decode(latent, decoder_input_ids)

    def generate(self, latent: Tensor, start_token_id: int, end_token_id: int, max_length: int) -> Tensor:
        """Greedily decode token ids from latent vectors."""
        batch_size = latent.size(0)
        current_tokens = torch.full(
            (batch_size, 1),
            start_token_id,
            dtype=torch.long,
            device=latent.device,
        )
        decoder_hidden = self.from_latent(latent).unsqueeze(0)
        generated_tokens = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=latent.device)

        for _ in range(max_length):
            embedded = self.embedding(current_tokens[:, -1:])
            outputs, decoder_hidden = self.decoder(embedded, decoder_hidden)
            logits = self.output_layer(outputs[:, -1, :])
            next_tokens = logits.argmax(dim=-1)
            generated_tokens.append(next_tokens)
            current_tokens = next_tokens.unsqueeze(1)
            finished |= next_tokens.eq(end_token_id)
            if bool(finished.all()):
                break

        if not generated_tokens:
            return torch.empty((batch_size, 0), dtype=torch.long, device=latent.device)

        return torch.stack(generated_tokens, dim=1)
