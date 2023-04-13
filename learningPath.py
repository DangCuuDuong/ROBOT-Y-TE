#!/usr/bin/env python2
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
import math
from geometry_msgs.msg import PoseStamped
import message_filters
from tf.transformations import euler_from_quaternion, quaternion_from_euler
x = []
y = []
trajec = [[0,0,0]]
counter = 0
def getdata(rp,pose):
    
    a = pose.pose
    rot = [a.orientation.x,a.orientation.y,a.orientation.z,a.orientation.w]
    (_, _, yaw) = euler_from_quaternion (rot)

    rotation =math.degrees(yaw)

    xPose = a.position.y * 100
    yPose = a.position.x * -100
    dis = ( (xPose - trajec[-1][0] )**2 + ( yPose - trajec[-1][1] )**2 ) ** 0.5
    if dis >= 50:
        trajec.append([xPose,yPose,rotation])
    print(round(xPose),round(yPose))
    np.savetxt('/home/phuong/catkin_ws/src/beginner_tutorials/scripts/trajec.txt',trajec)


rospy.init_node("plojhggtter")

rp = message_filters.Subscriber("scan",LaserScan)
pose = message_filters.Subscriber("slam_out_pose", PoseStamped)
ts = message_filters.TimeSynchronizer([rp,pose],100)
ts.registerCallback(getdata)
rospy.spin()



    