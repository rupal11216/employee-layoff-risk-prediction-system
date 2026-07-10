import torch
import torch.nn as nn

class LayoffRNN(nn.Module):
    """
    Simple LSTM-based classifier producing one logit per sample.
    Input shape: (batch, seq_len, features)
    Output: (batch, 1) logits
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, (hn, cn) = self.lstm(x)        # out: (batch, seq_len, hidden)
        last = out[:, -1, :]                # take last time-step
        logit = self.fc(last)               # (batch, 1)
        return logit
