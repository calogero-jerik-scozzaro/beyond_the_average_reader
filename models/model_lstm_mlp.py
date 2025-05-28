import torch
import torch.nn as nn

class MyLSTMMLP(nn.Module):
    def __init__(self):
        super(MyLSTMMLP, self).__init__()
        self.lstm = nn.LSTM(
            input_size=20, #
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        # Define the MLP (1-layer)
        self.mlp = nn.Sequential(
            nn.Linear(64, 1), # now it has to be 1, and modify the forward function
            nn.Tanh()
        )

    def forward(self, x):
        # Pass input through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch_size, sequence_length, hidden_size]
        
        # Apply the MLP to each timestep output from LSTM
        # lstm_out is [batch_size, sequence_length, hidden_size]
        # We will apply the MLP to each time step independently, so we need to reshape for that
        batch_size, seq_len, hidden_size = lstm_out.shape
        
        # Reshape the LSTM output to [batch_size * seq_len, hidden_size]
        lstm_out = lstm_out.contiguous().view(batch_size * seq_len, hidden_size)  # Flatten the sequence dimension
        
        # Pass the reshaped tensor through MLP
        mlp_out = self.mlp(lstm_out)  # [batch_size * seq_len, 1]
        
        # Reshape the output back to [batch_size, sequence_length, 1]
        mlp_out = mlp_out.view(batch_size, seq_len, 1)
        
        return mlp_out