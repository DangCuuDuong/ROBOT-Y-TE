#! /usr/bin/env python2
import matplotlib.pyplot as plt
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import message_filters

import math
import cv2
import serial
import time
rospy.init_node("plotter")
pub = rospy.Publisher('move', String, queue_size=10)
def callback(msg1):
    print("a")
    pub.publish("d")

rospy.Subscriber("/scan",LaserScan,callback)
rospy.spin()

    