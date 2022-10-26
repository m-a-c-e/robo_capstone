#!/usr/bin/env python
import copy
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3

class TurtleBot:
    def __init__(self):
        velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)



