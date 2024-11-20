import torch.nn as nn

class ResponseTimePredictor(nn.Module):
    def __init__(self, embedding_dim):
        super(ResponseTimePredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predict a single scalar value (response time)
        )

    def forward(self, x):
        return self.mlp(x)
