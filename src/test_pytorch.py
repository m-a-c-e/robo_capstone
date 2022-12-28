#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        output = F.softmax(output, dim=-1)
        return output

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    ac = Actor(360, 11)
    ac = ac.to('cuda:0')
    ac = ac.to(torch.float64)
    lidar = torch.arange(360)
    lidar = lidar.to(torch.float64)
    lidar = torch.unsqueeze(lidar, dim=0)

    ans = ac(lidar)
    print(torch.sum(ans))
