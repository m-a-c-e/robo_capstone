#!/usr/bin/env python
import copy
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan


class TurtleBot:
    def __init__(self):
        rospy.init_node('generate_rollout', anonymous=True)     # initialize the node with filename
        rospy.Subscriber('/scan', LaserScan, self.get_lidar_scan, queue_size=1)
        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1) 		
        self.lidar_scan = []
        self.rate = rospy.Rate(1)                                    # 1 Hz 
        self.state = None
        self.policy = np.array([0.5, 0.5]) 
        self.collision = False
        self.actions = np.arange([1, -1])
        self.n_actions = 2
        self.velocity = Twist()
        self.states = [get_lidar_scan]

    def get_lidar_scan(self, data):
        # ranges = data.ranges
        self.lidar_scan = np.array(data.ranges)

    def set_velocity(self, v_in, w_in):
        # set the linear and/or angular velocity
        if v_in is not None:
            self.velocity.linear = Vector(v_in[0], v_in[1], v_in[2])
        if w_in is not None:
            self.velocity.angular = Vector(w_in[0], w_in[1], w_in[2])
        self.velocity_pub.publish(self.velocity)



if __name__ == "__main__":
    print("Starting simulation....")
    # start simulation
    # pause simulation

    # set the constant linear velocity
    tb.set_velocity([0, 5, 0], None)

    print("Generating rollout....")
    tb = TurtleBot()                                                # initialize turtlebot object 
    t_i = 0

    while not rospy.is_shutdown():
        action_i = np.random.choice(np.arange(tb.n_actions), 1, self.policy)[0]  # probabilistically sample action. Takes care of exploration
        
        # set velocity
        tb.set_velocity(None, [0, 0, action_i])

        # play simulation
        # 

        # pause simulation

        # reset the velocity
        tb.set_velocity(None, [0, 0, 0])

        # get the new stat 
        tb.states.append(tb.get_lidar_scan)

        rate.sleep() # to run at a desired rate
        time_step += 1
        if time_step == 10 or self.collision:
            # if max rollout length achieved
            break
    print("....closing Generating rollout")
