#!/usr/bin/env python
import copy
import rospy
import numpy as np
import time
from sensor_msgs.msg import LaserScan


class TurtleBot:
    def __init__(self):
        rospy.Subscriber('/scan', LaserScan, self.get_lidar_scan, queue_size=1)
        self.lidar = []

    def get_lidar_scan(self, data):
        ranges = np.clip(np.array(data.ranges), 0, 5)
        self.lidar = np.array(ranges)  # contains 360 dists (in meters) {0, 1, ... , 359}

if __name__ == "__main__":
    print("start")
    rospy.init_node('test_lidar', anonymous=True)
    tb = TurtleBot()
    time.sleep(1)
    print(tb.lidar)
    while not rospy.is_shutdown():
        print("front = ", np.round(tb.lidar[0], 2))
        print("left = ", np.round(tb.lidar[90], 2))
        print("back = ", np.round(tb.lidar[180], 2))
        print("right = ", np.round(tb.lidar[275], 2))
        time.sleep(3)
    print("end")
