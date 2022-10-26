#!/usr/bin/env python
import rospy
import os


if __name__ == "__main__":
    print("Starting simulation....")
    os.system("robocap_launch_sim")
    print("Shutting down node....")

