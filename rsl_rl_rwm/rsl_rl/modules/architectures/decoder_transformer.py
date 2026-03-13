import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 256, device: str = "cpu"):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DecoderTransformerBase(nn.Module):
    """Causal (decoder-only) transformer with sinusoidal positional encoding.

    Matches the RNNBase / MLPBase interface expected by SystemDynamicsEnsemble:
      - forward(x_state_batch, x_action_batch) -> (batch, hidden_size)
      - reset() / reset_partial()  -- stateless, both are no-ops

    architecture_config keys:
        decoder_transformer_d_model     (int):   model dimension        (default 64)
        decoder_transformer_num_heads   (int):   number of attention heads (default 8)
        decoder_transformer_num_layers  (int):   number of decoder layers  (default 2)
        decoder_transformer_ffn_dim     (int):   feedforward hidden dim    (default 256)
        decoder_transformer_dropout     (float): dropout probability       (default 0.0)
        decoder_transformer_max_seq_len (int):   maximum sequence length   (default 32)
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
        d_model = cfg.get("decoder_transformer_d_model", 64)
        num_heads = cfg.get("decoder_transformer_num_heads", 8)
        num_layers = cfg.get("decoder_transformer_num_layers", 2)
        ffn_dim = cfg.get("decoder_transformer_ffn_dim", 256)
        dropout = cfg.get("decoder_transformer_dropout", 0.0)
        max_seq_len = cfg.get("decoder_transformer_max_seq_len", 32)

        self.hidden_size = d_model

        self.input_proj = nn.Linear(input_dim, d_model, device=device)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len, device=device)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            device=device,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self._causal_mask: torch.Tensor | None = None
        self._causal_mask_len: int = 0

    def forward(self, x_state_batch: torch.Tensor, x_action_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_state_batch:  (batch, seq_len, state_dim)
            x_action_batch: (batch, seq_len, action_dim)

        Returns:
            (batch, hidden_size) — representation of the last token.
        """

        x = torch.cat([x_state_batch, x_action_batch], dim=-1)  # (B, T, input_dim)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_encoding(x)  # (B, T, d_model)

        mask = self._get_causal_mask(x.size(1), x.device)
        x = self.decoder(x, mask=mask, is_causal=True)  # (B, T, d_model)

        return x[:, -1]  # (B, hidden_size)

    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask is None or self._causal_mask_len != seq_len:
            self._causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
            self._causal_mask_len = seq_len
        return self._causal_mask
