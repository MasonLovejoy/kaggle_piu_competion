# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:32:01 2024

@author: Mason
"""
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(55, 2048, dtype = torch.float64),
            nn.ReLU(),
            nn.Linear(2048, 2048, dtype = torch.float64),
            nn.ReLU(),
            nn.Linear(2048, 2048, dtype = torch.float64),
            nn.ReLU(),
            nn.Linear(2048, 4, dtype = torch.float64),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits