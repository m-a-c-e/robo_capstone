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
        lin_out = torch.unsqueeze((output[0] + 1) / 20, dim=0)
        ang_out = torch.unsqueeze(output[1] / 2, dim=0)
        return lin_out, ang_out


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

        self.num_actions      = 2
            

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
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            pass
        else:
            print("loading saved model...")
            checkpoint = torch.load(self.load_model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
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

    def get_reward(self, linear_vel):
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

        lin_reward = linear_vel * 10
        if fwd_lidar >= -1:
            reward += self.p_reward
        else:
            reward += self.n_reward

            lin_reward += self.n_reward
            terminate = True

        return reward, terminate, lin_reward

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

        st = 0
        time_list = []

        for t in range(self.max_time_steps):

            self.optimizer.zero_grad()
            
            # state
            state = torch.from_numpy(self.lidar).to(torch.float64)
            state = (state - 0.15) / (3.5 - 0.15)

            # actions
            #action_mean = self.model(state)
            lin_mean, ang_mean = self.model(state)

            # sample actions
            #action = torch.normal(action_mean, self.sigma)  # gaussian sampling
            #action = torch.clamp(action, min=0, max=0.1)
            lin_action = torch.normal(lin_mean, self.sigma / 10)  # gaussian sampling
            lin_action = torch.clamp(lin_action, min=0, max=0.1)
            ang_action = torch.normal(ang_mean, self.sigma)  # gaussian sampling
            ang_action = torch.clamp(ang_action, min=-0.5, max=0.5)



            # take action in the simulation
            self.set_velocity([lin_action.data, 0, 0],[0, 0, ang_action.data]) 

            # os.system("rosservice call /gazebo/unpause_physics")
            # os.system("gz world --step")
            time.sleep(0.0213)      # for 5.5x speed up in gazebo
            # os.system("rosservice call /gazebo/pause_physics")

            # collect reward from taking the action
            reward, terminate, lin_reward = self.get_reward(lin_action) # in all other cases
            reward_rollout.append([lin_reward, reward])

            # store probability of taking that action
            #prob   = 1 / (4.443 * self.sigma * torch.exp((action - action_mean) ** 2 / (2 * self.sigma) ** 2)) 

            lin_prob = 1 / (4.443 * self.sigma * torch.exp((lin_action - lin_mean) ** 2 / (2 * self.sigma) ** 2)) 

            ang_prob = 1 / (4.443 * self.sigma * torch.exp((ang_action - ang_mean) ** 2 / (2 * self.sigma) ** 2)) 

            prob = torch.cat((lin_prob, ang_prob))
            prob = torch.unsqueeze(prob, dim=0)
            prob_rollout.append(prob)

            # decide whether to end rollout or not
            if terminate:
                break

        # normalize rewards between 0 and 1
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
    json_file.close()

    tb = TurtleBot(args_dict)

    ### need some time to initialize the lidar readings
    time.sleep(1)
    ############################

    iterations = 20000

    devisor = iterations / 4

    sigma_schedular = np.arange(0.01, tb.sigma + 0.1, (tb.sigma + 0.1 - 0.01) / 4)
    sigma_schedular = np.flip(sigma_schedular)

    state_rollout = None
    action_rollout = None
    reward_rollout = None
    gamma = torch.tensor(0.99, dtype=torch.float64, requires_grad=False)

    while not rospy.is_shutdown ():
        for i in range(iterations):
            # sigma setter
            sigma_idx = i // devisor
            tb.sigma = sigma_schedular[sigma_idx]

            # 1. generate rollout
            prob_rollout, reward_rollout = tb.generate_rollout()
            cum_reward_rollout = torch.empty(reward_rollout.size(), requires_grad=False, dtype=torch.float64)

            # 2. calculate expected cummulative reward
            T_terminal = reward_rollout.size()[0]
            for j in range(T_terminal):
                cum_reward_lin = 0
                cum_reward_ang = 0
                for k in range(j, T_terminal):
                    diff = torch.tensor(k - j, dtype=torch.float64, requires_grad=False)
                    cum_reward_lin += torch.pow(gamma, diff) * reward_rollout[j][0]
                    cum_reward_ang += torch.pow(gamma, diff) * reward_rollout[j][1]

                cum_reward_rollout[j][0] = cum_reward_lin
                cum_reward_rollout[j][1] = cum_reward_ang

            # 3. multiply reward with the log probs
            loss = cum_reward_rollout * torch.log(prob_rollout)
            loss = torch.mean(loss)
            loss = -1 * loss # for gradient ascent
            loss.backward()
            tb.optimizer.step()
            
            # 4. print metrics
            print("Iteration # : {} Mean Reward: {}  Timesteps: {}".format(i, torch.mean(reward_rollout).data, T_terminal))
            if i % 100 == 0: 
                now = datetime.now()
                date_string = now.strftime("%d-%m-%Y/")
                time_string = now.strftime("%H-%M-%S_")
                dir_path = "/home/manan/catkin_ws/src/robo_capstone/src/trained_models/" + date_string
                if os.path.exists(dir_path):
                    pass
                else:
                    os.makedirs(dir_path)
                tb.args_dict['load_model'] = dir_path + time_string + str(i) + ".pt"

                # torch.save(tb.model, dir_path + time_string + str(i) + ".pt")

                torch.save({
                    'model_state_dict': tb.model.state_dict(),
                    'optimizer_state_dict': tb.optimizer.state_dict()
                    }, dir_path + time_string + str(i) + ".pt")

                args_file = open(dir_path + time_string + str(i) + ".json", 'w')
                args_json = json.dumps(tb.args_dict)
                args_file.write(args_json)
                args_file.close()
        break
    os.system("rosservice call /gazebo/reset_simulation")    
