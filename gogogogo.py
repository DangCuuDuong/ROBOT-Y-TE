#!/usr/bin/env python2
from matplotlib.pyplot import ylabel
import numpy as np
import rospy
from rospy.topics import Publisher
from sensor_msgs.msg import LaserScan
import math
from geometry_msgs.msg import PoseStamped
import message_filters
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String
pub = Publisher('move',String,queue_size=10)
trajec = []
i = 0
counter = 0
avoid = False
prevP = []

def calDis(x1,x2,y1,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def calAngle(x,y):
    vector_1 = [x,y]
    vector_2 = [0, 1]

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = math.degrees(np.arccos(dot_product))
    return angle
def getdata(rp,pose):
    global counter
    a = pose.pose
    rot = [a.orientation.x,a.orientation.y,a.orientation.z,a.orientation.w]
    (_, _, yaw) = euler_from_quaternion (rot)
    rotation =math.degrees(yaw)
    if rotation < 0:
        rotation = rotation + 360
    rotation = 360 - rotation
        
    xPose = a.position.y * 100
    yPose = a.position.x * -100
    global i
    i += 1
    a = calAngle(-1 *xPose,-1*yPose)

    if rp.ranges[0]*100>100 and rp.ranges[5] * 100 > 100:
        if i % 7 == 0:
            pub.publish("w")
            # counter += 1
            prevP.append([xPose,yPose])
            i = 0
    elif rp.ranges[90] * 100 > 40 or rp.ranges[85] * 100 > 40:
        if i % 24 == 0:
            counter = 0
            pub.publish("d")
            pub.publish("d")
            pub.publish("d")
            pub.publish("d")
            pub.publish("d")
            pub.publish("d")
            i = 0
    print("................................................")
    print("Robot at x = %f y = %f"%(xPose,yPose))
    print("Distance from R to (0,0): %f"%calDis(xPose,0,yPose,0))
    
if __name__ == '__main__':

    trajec = np.loadtxt('/home/phuong/catkin_ws/src/beginner_tutorials/trajec.txt')
    rospy.init_node("plottasder")
    rp = message_filters.Subscriber("scan",LaserScan)
    pose = message_filters.Subscriber("slam_out_pose", PoseStamped)
    ts = message_filters.ApproximateTimeSynchronizer([rp, pose], 10, 0.1, allow_headerless=True)

    ts.registerCallback(getdata)
    rospy.spin()



    