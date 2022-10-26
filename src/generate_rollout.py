#!/usr/bin/env python
import copy
import rospy

import os
import time
import numpy as np

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3


class TurtleBot:
    def __init__(self):
        # nodes, subscribers and publishers
        rospy.init_node('test_pause_play', anonymous=True)      # initialize the node with filename
        rospy.Subscriber('/scan', LaserScan, self.get_lidar, queue_size=1)	        
        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)	

        self.velocity         = Twist()
        self.velocity.linear  = Vector3(0, 0, 0)
        self.velocity.angular = Vector3(0, 0, 0)

        self.t_max            = 5 # seconds
        self.actions          = np.round(np.arange(-1, 1, 0.2), 2)
        self.lidar            = None

        self.policy           = 1 / 11 * np.ones(11)

        self.wall_dist        = 0.4 # meters
            
        self.model            = None


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
        reward = -1 * np.mean(weighted_lidar)
        return reward


if __name__ == "__main__":
    tb = TurtleBot()
    time.sleep(1)
    i = 0
    rollout = []
    reward = None
    time_start = None
    time_end = None
    while not rospy.is_shutdown():
        time_start = time.time()
        if i == 10:
            break
        if i != 0:
            # get the reward
            reward = tb.get_reward()
            rollout.append(reward)
        #os.system("rosservice call /gazebo/pause_physics")

        # Sample Action (uniform currently, set the probability distribution using model)
        at = np.random.choice(tb.actions, 1) 
        
        # take action in the simulation for 1 second
        #os.system("rosservice call /gazebo/unpause_physics")

        #time.sleep(0.1)
        i += 1
        time_end =time.time()
        if i <= 10:
            print("iteration {} = {}".format(i, time_end - time_start))
    # reset simulation once
    os.system("rosservice call /gazebo/reset_simulation")
