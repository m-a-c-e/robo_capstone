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
        #out1 = torch.unsqueeze(output[0][0])
        return output

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
    #print(lidar1.size())

    # output
    out = []
    out1 = ac(lidar1)
    
    lin = torch.unsqueeze(out1[0][0], dim=0)
    lin = lin / 2

    ang = torch.unsqueeze(out1[0][1], dim=0)
    ang = (ang + 1) / 20

    out.append(lin)
    out.append(ang)
    out = torch.cat(out)
    print(out.size())

    # gradient check
    print(ac.linear1.weight.grad)
    loss = torch.mean(out)
    print(loss)
    loss.backward()
    print(ac.linear1.weight.grad)
    
    exit()

    out = torch.cat(out)

    # prob
    print(ac.linear1.weight.grad)
    loss = torch.mean(out)
    print(loss)
    loss.backward()
    print(ac.linear1.weight.grad)

