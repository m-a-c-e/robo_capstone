#!/usr/bin/env python
import copy
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
import subprocess
import os

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3


class TurtleBot:
    def __init__(self):
        rospy.init_node('test_set_velocity', anonymous=True)     # initialize the node with filename
        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1) 		
        self.velocity = Twist()
        self.velocity.linear = Vector3(0, 0, 0)
        self.velocity.angular = Vector3(0, 0, 0)

    def set_velocity(self, v_in, w_in):
        # set the linear and/or angular velocity
        if v_in is not None:
            self.velocity.linear = Vector3(v_in[0], v_in[1], v_in[2])
        if w_in is not None:
            self.velocity.angular = Vector3(w_in[0], w_in[1], w_in[2])
        self.velocity_pub.publish(self.velocity)



if __name__ == "__main__":
    tb = TurtleBot()
    while not rospy.is_shutdown():
        tb.set_velocity([0.1, 0, 0], None)
        pass
