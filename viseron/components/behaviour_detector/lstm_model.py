"""LSTM model for behaviour detection."""
from torch import nn


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for behaviour detection."""

    def __init__(
        self, input_dim=512, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.3
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass of the model."""
        # x: [B, T, 512]
        lstm_out, _ = self.lstm(x)  # [B, T, 2*hidden_dim]
        out = lstm_out[:, -1, :]  # use last time step
        out = self.dropout(out)
        out = self.fc(out)
        return out
