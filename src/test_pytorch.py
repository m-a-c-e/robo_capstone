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
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        output = torch.tanh(output)
        return output

if __name__ == "__main__":
    print("pytorch version: ", torch.__version__)
    prob_rollout   = []
    state_rollout = []

    ac = Actor(360, 1)
    ac = ac.to(torch.float64)

    lidar1 = torch.arange(360)
    lidar2 = torch.arange(360) / 10
    lidar1 = torch.unsqueeze(lidar1.to(torch.float64), dim=0)
    lidar2 = torch.unsqueeze(lidar2.to(torch.float64), dim=0)

    # input
    state_rollout = torch.cat((lidar1, lidar2), dim=0)
    print(state_rollout.size())

    # output
    action_rollout = ac.forward(state_rollout)
    action_rollout = torch.clamp(action_rollout, min=-0.5, max=0.5)
    print(action_rollout.size())

    # prob
    prob_rollout = action_rollout / 2
    print(prob_rollout.size())

    print(ac.linear1.weight.grad)
    loss = torch.mean(prob_rollout)
    loss.backward()
    print(ac.linear1.weight.grad)

