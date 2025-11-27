import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)  # ADD THIS
        )
    def forward(self, x):
        return self.net(x)