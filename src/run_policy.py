#!/usr/bin/env python
import copy
import rospy

import os
import time
import numpy as np
import torch
import math

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import keyboard
from datetime import datetime
import json
import sys

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
        output = output / 2
        return output


class TurtleBot:
    def __init__(self, args_dict):
        # nodes, subscribers and publishers
        rospy.init_node('test_pause_play', anonymous=True)      # initialize the node with filename
        rospy.Subscriber('/scan', LaserScan, self.get_lidar, queue_size=1)	        
        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)	

        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.update_Odometry) # handle to get the position

        self.Init = True    # only for when the robot publishes the first odometry data
        self.Init_ang = 0 
        self.Init_pos = Vector3(0, 0, 0)	
        self.globalPos = Vector3(0, 0, 0)
        self.globlaAng = 0

        self.velocity         = Twist()
        self.velocity.linear  = Vector3(0, 0, 0)
        self.velocity.angular = Vector3(0, 0, 0)

        self.lidar            = np.zeros(360)

        self.num_actions      = 1
            

        # params
        self.args_dict = args_dict
        print(args_dict)
        self.sigma = args_dict['sigma']
        self.lidar_start = args_dict['lidar_start']
        self.lidar_end = args_dict['lidar_end']
        self.cnst_vel = args_dict['cnst_vel']
        self.allowed_error = args_dict['allowed_error']
        self.p_reward = args_dict['p_reward']
        self.n_reward = args_dict['n_reward']
        self.max_time_steps = args_dict['max_time_steps']
        self.load_model = args_dict['load_model']
        self.wall_dist = args_dict['wall_dist']
       
        self.model = Actor(self.lidar.size, self.num_actions).to(torch.float64)
        self.optimizer = None

        if self.load_model == '':
            print("Initialising model...")
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            pass
        else:
            print("loading saved model...")
            checkpoint = torch.load(self.load_model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def set_velocity(self, v_in, w_in):
        # set the linear and/or angular velocity
        if v_in is not None:
            self.velocity.linear  = Vector3(v_in[0], v_in[1], v_in[2])
        if w_in is not None:
            self.velocity.angular = Vector3(w_in[0], w_in[1], w_in[2])
        self.velocity_pub.publish(self.velocity)

    def get_lidar(self, data):
        ranges = np.clip(np.array(data.ranges), 0.15, 3.5)     # clip between 0.3 and 5 meters
                                                            # due to measurement limiations
        self.lidar = np.array(ranges)  # contains 360 dists (in meters) {0, 1, ... , 359}

    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        if self.Init:
        #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.Init = False
            self.Init_ang = orientation			 				# radians
            self.globalAng = self.Init_ang						# radians
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y				# meters
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y				# meters
            self.Init_pos.z = position.z		
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x		# meters
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y		# meters
        self.globalAng = orientation - self.Init_ang			# radians



    def get_reward(self):
        curr_lidar = np.array(self.lidar)
        left_lidar = np.array(curr_lidar[self.lidar_start:self.lidar_end])
        ang_comp = np.cos(np.arange(-2, 3, 1) * np.pi / 180)
        left_lidar = left_lidar * ang_comp
        left_lidar = np.abs(left_lidar - self.wall_dist) # get the distance to wall
        dist_mu = np.mean(left_lidar)
        reward = 0
        terminate = False

        # reward for staying parallel to the wall
        if dist_mu < self.allowed_error:
            reward += self.p_reward
        else:
            reward += self.n_reward
            terminate = True

        # penalty for not avoiding the wall
        fwd_lidar = np.array([curr_lidar[1], curr_lidar[0], curr_lidar[359]])
        fwd_lidar -= self.wall_dist
        fwd_lidar = np.where(fwd_lidar < 0, -1, 0)

        fwd_lidar = np.sum(fwd_lidar)
        
        if fwd_lidar >= -1:
            reward += self.p_reward
        else:
            reward += self.n_reward
            terminate = True

        return reward, terminate

    def generate_rollout(self):
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
        startx = 0
        starty = 0
        dist   = 0

        for t in range(self.max_time_steps):
            self.optimizer.zero_grad()

            # state
            state = torch.from_numpy(self.lidar).to(torch.float64)
            state = (state - 0.15) / (3.5 - 0.15)

            # action
            action_mean = self.model(state)
            action = torch.normal(action_mean, self.sigma)
            prob   = 1 / (4.443 * self.sigma * torch.exp((action - action_mean) ** 2 / (2 * self.sigma) ** 2)) 

            # take action in the simulation
            self.set_velocity([self.cnst_vel, 0, 0],[0, 0, action.data]) 

            # os.system("rosservice call /gazebo/unpause_physics")
            # os.system("gz world --step")
            # os.system("rosservice call /gazebo/pause_physics")
            # time step ~ 1.08 seconds
            time.sleep(1)

            # collect reward from taking the action
            reward, terminate = self.get_reward() # in all other cases

            # store reward and probabilities for each time step
            reward_rollout.append(reward)
            prob_rollout.append(prob)

            # decide whether to end rollout or not
            if terminate:
                break

        # normalise rewards between 0 and 1
        reward_rollout = torch.tensor(reward_rollout, requires_grad=False, dtype=torch.float64)
        prob_rollout = torch.cat(prob_rollout, dim=0)

        # reset simulation once
        os.system("rosservice call /gazebo/reset_simulation")    
        return (prob_rollout, reward_rollout)



if __name__ == "__main__":
    args = sys.argv
    json_file = args[1]
    json_file = open(json_file,"r")
    args_dict = json.load(json_file)

    tb = TurtleBot(args_dict)
    tb.model.eval()

    ### need some time to initialize the lidar readings
    time.sleep(1)
    ############################

    startx = 0
    starty = 0
    dist = 0

    while not rospy.is_shutdown ():
        with torch.no_grad():
            state = torch.from_numpy(tb.lidar).to(torch.float64)
            state = (state - 0.15) / (3.5 - 0.15)
            action_mean = tb.model.forward(state)
            tb.set_velocity([tb.cnst_vel, 0, 0],[0, 0, action_mean.data]) 
            print(action_mean.data)

            time.sleep(1)
    os.system("rosservice call /gazebo/reset_simulation")    
