import torch.nn as nn
import torch

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


class QueryTimePredictor(nn.Module):
    def __init__(self, rnn_hidden_dim, embedding_dim):
        super(QueryTimePredictor, self).__init__()

        # RNN for processing (query_embed, time_spent) pairs
        self.rnn = nn.LSTM(input_size=embedding_dim + 1,  # query_embed + time_spent
                           hidden_size=rnn_hidden_dim,
                           num_layers=1,
                           batch_first=True)

        # Fully connected layers for prediction
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden_dim + embedding_dim, 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, query_embeds, time_spents, current_query_embed):
        rnn_input = torch.cat([query_embeds, time_spents], dim=-1)  # (batch_size, seq_len, embed_dim + 1)

        rnn_output, _ = self.rnn(rnn_input)  # rnn_output: (batch_size, seq_len, rnn_hidden_dim)
        last_rnn_output = rnn_output[:, -1, :]  # Get last output of the RNN (batch_size, rnn_hidden_dim)

        combined = torch.cat([last_rnn_output, current_query_embed], dim=-1)  # (batch_size, rnn_hidden_dim + embed_dim)
        prediction = self.fc(combined)  # (batch_size, output_dim)

        return prediction
