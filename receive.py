#!/usr/bin/env python2
import rospy
from std_msgs.msg import String
import numpy as np

import serial



ser = serial.Serial('/dev/ttyUSB1', 115200)
def callback(data):    
    print(data.data)
    a = data.data
    if a == "a":
        a = "l"
    elif a == "d":
        a ="r"
    elif a =="w":
        a = "f"
    elif a == "s":
        a = "b"
    elif a == "m":
        rospy.signal_shutdown("cannot move")
    a = str.encode(a)
    ser.write(a)

    

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/move", String, callback)
    # rospy.Subscriber("/camera/rgb/image_color", sensor_msgs.msg.Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
    
