#!/usr/bin/env python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable


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
        lin_out = torch.unsqueeze((output[0][0] + 1) / 20, dim=0)
        ang_out = torch.unsqueeze(output[0][1] / 2, dim=0)

        return lin_out, ang_out

if __name__ == "__main__":
    print("pytorch version: ", torch.__version__)
    prob_rollout   = []
    state_rollout = []

    ac = Actor(360, 2)
    ac = ac.to(torch.float64)

    # input
    lidar1 = torch.arange(360)
    lidar2 = torch.arange(360) / 10
    lidar1 = torch.unsqueeze(lidar1.to(torch.float64), dim=0)
    lidar2 = torch.unsqueeze(lidar2.to(torch.float64), dim=0)

    # output
    out = []
    lin_out, ang_out = ac(lidar1)
    actions = torch.cat((lin_out, ang_out))
    actions = torch.unsqueeze(actions, dim=0)
    out.append(actions)

    lin_out, ang_out = ac(lidar2)
    actions = torch.cat((lin_out, ang_out))
    actions = torch.unsqueeze(actions, dim=0)
    out.append(actions)

    out = torch.cat(out, dim=0)

    print(out.size())
    
    # gradient check
    print(ac.linear1.weight.grad)
    loss = torch.mean(out)
    loss.backward()
    print(ac.linear1.weight.grad)
