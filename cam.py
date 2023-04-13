#!/usr/bin/env python2
import rospy
import numpy as np
import sensor_msgs.msg._Image
import cv2
import serial
from cv_bridge import CvBridge
from cv_bridge.boost.cv_bridge_boost import getCvType
from PIL import Image, ImageTk

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
bridge = CvBridge()
img = 0

def callback(data):

    img = bridge.imgmsg_to_cv2(data)
    cv2.imshow('',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("hello")

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/kinect2/hd/image_color", sensor_msgs.msg.Image, callback)
    # rospy.Subscriber("/camera/rgb/image_color", sensor_msgs.msg.Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
    
