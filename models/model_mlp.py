import torch
import torch.nn as nn
import torch.optim as optim

class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(20, 10),
                                nn.Sigmoid(),
                                nn.Linear(10, 10),
                                nn.Sigmoid(),
                                nn.Linear(10, 1)
                                )
        
    def forward(self, x):
        return self.mlp(x)