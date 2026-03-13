import math

import torch
import torch.nn as nn


class TransformerBase(nn.Module):
    """Causal transformer encoder over the history horizon.

    Matches the RNNBase / MLPBase interface expected by SystemDynamicsEnsemble:
      - forward(x_state_batch, x_action_batch) -> (batch, hidden_size)
      - reset() / reset_partial()  -- stateless, both are no-ops

    architecture_config keys:
        transformer_hidden_size (int): model dimension d_model (default 256)
        transformer_num_heads   (int): number of attention heads (default 4)
        transformer_num_layers  (int): number of TransformerEncoderLayer blocks (default 2)
        transformer_ffn_dim     (int): feedforward hidden dim (default 512)
        transformer_dropout     (float): dropout probability (default 0.0)
        transformer_max_seq_len (int): maximum sequence length for positional encoding (default 256)
    """

    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device

        cfg = architecture_config or {}
        hidden_size = cfg.get("transformer_hidden_size", 256)
        num_heads = cfg.get("transformer_num_heads", 4)
        num_layers = cfg.get("transformer_num_layers", 2)
        ffn_dim = cfg.get("transformer_ffn_dim", 512)
        dropout = cfg.get("transformer_dropout", 0.0)
        max_seq_len = cfg.get("transformer_max_seq_len", 256)

        self.hidden_size = hidden_size

        # Project raw (state, action) tokens into d_model
        self.input_proj = nn.Linear(input_dim, hidden_size, device=device)

        # Learned positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size, device=device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            device=device,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Causal mask is built lazily and cached
        self._causal_mask: torch.Tensor | None = None
        self._causal_mask_len: int = 0

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def forward(self, x_state_batch: torch.Tensor, x_action_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_state_batch:  (batch, seq_len, state_dim)
            x_action_batch: (batch, seq_len, action_dim)

        Returns:
            (batch, hidden_size) — representation of the last token.
        """
        x = torch.cat([x_state_batch, x_action_batch], dim=-1)  # (B, T, input_dim)
        B, T, _ = x.shape

        x = self.input_proj(x)  # (B, T, hidden_size)

        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embedding(positions).unsqueeze(0)  # (B, T, hidden_size)

        mask = self._get_causal_mask(T, x.device)
        x = self.transformer(x, mask=mask, is_causal=True)  # (B, T, hidden_size)

        return x[:, -1]  # (B, hidden_size)

    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask_len != seq_len:
            self._causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            self._causal_mask_len = seq_len
        return self._causal_mask
