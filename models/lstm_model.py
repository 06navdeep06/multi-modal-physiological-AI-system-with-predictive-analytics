"""LSTM Stress Predictor

Bidirectional LSTM for sequence-based stress classification.
Fixed: hidden_size was previously hardcoded as 32 in forward() instead of
using self.hidden_size, which caused shape mismatches with any other value.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """Bidirectional LSTM for temporal stress prediction.

    Args:
        input_size:  Feature dimension per timestep (matches FeatureFusion.feature_dim)
        hidden_size: LSTM hidden units (default 64)
        num_layers:  Stacked LSTM layers (default 2)
        num_classes: Output classes – 3 for Low / Medium / High stress
        dropout:     Dropout probability between LSTM layers (default 0.3)
        bidirectional: Use bidirectional LSTM (default True)
    """

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 3,
                 dropout: float = 0.3, bidirectional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        # Classifier on concatenated last hidden state (both directions)
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_size)

        Returns:
            Tensor of shape (batch, num_classes) – raw logits
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * self.num_directions,
                          batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(x, (h0, c0))      # (batch, seq_len, hidden*dirs)
        last = out[:, -1, :]                   # take final timestep
        last = self.dropout(last)
        return self.fc(last)                   # (batch, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method returning softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)
