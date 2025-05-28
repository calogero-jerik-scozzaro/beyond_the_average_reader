import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import torch.nn as nn
import torch.optim as optim

class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=20,   
                            hidden_size=64,  
                            num_layers=2,
                            batch_first=True)
            
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return out, (h_n, c_n)
