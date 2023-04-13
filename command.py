#!/usr/bin/env python2
import rospy
from std_msgs.msg import String
import numpy as np
import os
def callback(data):    
    print(data.data)
    a = data.data
    if a == "1":
        print("123")
        os.system("rosrun beginner_tutorials sub.py ")
    
    

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/command", String, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
    
