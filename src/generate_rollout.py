#!/usr/bin/env python
import copy
import rospy

import os
import time
import numpy as np
import torch

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

from test_pytorch import Actor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import keyboard


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


class TurtleBot:
    def __init__(self):
        # nodes, subscribers and publishers
        rospy.init_node('test_pause_play', anonymous=True)      # initialize the node with filename
        rospy.Subscriber('/scan', LaserScan, self.get_lidar, queue_size=1)	        
        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)	

        self.velocity         = Twist()
        self.velocity.linear  = Vector3(0, 0, 0)
        self.velocity.angular = Vector3(0, 0, 0)

        self.lidar            = np.zeros(360)

        self.wall_dist        = 0.4 # meters

        self.num_actions      = 1
            
        self.model            = Actor(self.lidar.size, self.num_actions).to(torch.float64)
        self.optimizer        = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        self.sigma = 0.1

    def set_velocity(self, v_in, w_in):
        # set the linear and/or angular velocity
        if v_in is not None:
            self.velocity.linear  = Vector3(v_in[0], v_in[1], v_in[2])
        if w_in is not None:
            self.velocity.angular = Vector3(w_in[0], w_in[1], w_in[2])
        self.velocity_pub.publish(self.velocity)

    def get_lidar(self, data):
        ranges = np.clip(np.array(data.ranges), 0.3, 5)     # clip between 0.3 and 5 meters
                                                            # due to measurement limiations
        self.lidar = np.array(ranges)  # contains 360 dists (in meters) {0, 1, ... , 359}

    def get_reward(self):
        # calculates the reward based on the ldiar input
        # Plan to stay within 0.1 meters on the left side of the robot
        # choose array from 45 to 135 degree lidar readings and weight each with a gaussian
        curr_lidar = np.array(self.lidar[44:135])        # create a copy of 91 readings
        curr_lidar = np.abs(curr_lidar - self.wall_dist) # get the distance to wall

        # create gaussian weights
        indices    = np.expand_dims(np.arange(curr_lidar.size), axis=0)
        mu  = np.mean(indices, keepdims=True)
        sigma = np.std(indices, keepdims=True)

        wts = (0.4 * np.exp(-(0.5) * ((indices - mu) / (sigma)) ** 2))
        weighted_lidar = curr_lidar * wts
        reward = - np.mean(weighted_lidar)
        return reward

    def generate_rollout(self, max_time_steps):
        i = 0
        reward_rollout = []
        action_rollout = []
        prob_rollout   = []
        state_rollout  = []
        action_prob_rollout = []
        state          = None
        action         = None
        reward         = None
        time_start     = None
        time_end       = None
        time_step_list = []

        for t in range(max_time_steps):
            time_start = time.time()

            # state
            state = torch.from_numpy(self.lidar).to(torch.float64)

            # action
            action_mean = self.model.forward(state)
            action = torch.normal(action_mean, 0.1)
            prob   = 1 / (4.443 * self.sigma * torch.exp((action - action_mean) ** 2 / (2 * self.sigma) ** 2)) 

            # take action in the simulation
            self.set_velocity([0.01, 0, 0],[0, 0, action.data]) 
            #os.system("rosservice call /gazebo/unpause_physics")
            #os.system("gz world --step")
            #os.system("rosservice call /gazebo/pause_physics")
            # time step ~ 1.08 seconds
            time.sleep(1)

            # collect reward from taking the action
            reward = self.get_reward()

            # store reward and probabilities for each time step
            reward_rollout.append(reward)
            prob_rollout.append(prob)

            time_end = time.time()
            ts       = time_end - time_start
            time_step_list.append(ts)
            # print("Iteration {}, action {}".format(t, action.data))

        time_step_list = torch.mean(torch.tensor(time_step_list, dtype=torch.float64))
        reward_rollout = torch.unsqueeze(torch.tensor(reward_rollout, requires_grad=False, dtype=torch.float64), dim=1)

        # normalize rewards between 0 and 1
        prob_rollout   = torch.unsqueeze(torch.tensor(prob_rollout, requires_grad=True, dtype=torch.float64), dim=1)

        # reset simulation once
        os.system("rosservice call /gazebo/reset_simulation")    
        # return state, action, reward tuple
        return (prob_rollout, reward_rollout)

if __name__ == "__main__":
    tb = TurtleBot()

    ### need some time to initialize the lidar readings
    time.sleep(1)
    ############################
    iterations = 10
    state_rollout = None
    action_rollout = None
    reward_rollout = None
    gamma = torch.tensor(0.99, dtype=torch.float64, requires_grad=False)
    st = None
    for i in range(iterations):
        st = time.time()
        # 1. generate rollout
        prob_rollout, reward_rollout = tb.generate_rollout(2)
        cum_reward_rollout = torch.empty(reward_rollout.size(), requires_grad=False, dtype=torch.float64)
        
        # 2. calculate expected cummulative reward
        T_terminal = reward_rollout.size()[0]
        for j in range(T_terminal):
            cum_reward = 0
            for k in range(j, T_terminal):
                diff = torch.tensor(k - j, dtype=torch.float64, requires_grad=False)
                cum_reward += torch.pow(gamma, diff) * reward_rollout[j] 
            cum_reward_rollout[j] = cum_reward 

        # 3. multiply reward with the log probs
        loss = cum_reward_rollout * torch.log(prob_rollout)
        loss = torch.mean(loss, dim=0)
        loss = -1 * loss # for gradient ascent
        loss.backward()
        tb.optimizer.step()
        tb.optimizer.zero_grad()
        et = time.time()
        
        # 4. print metrics
        print("Iteration # : {}    Mean Reward: {}  Time: {}".format(i, torch.mean(reward_rollout).data, round(et - st, 2)))
    torch.save(tb.model.state_dict(), "/home/mace/catkin_ws/src/robo_capstone/src/trained_models/test_model.pt")
    os.system("rosservice call /gazebo/reset_simulation")    

