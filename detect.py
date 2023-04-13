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
import serial
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
    
    # sensor = np.array(rp.ranges) * 100
    # k = 0
    # points = []
    # for i in sensor:
    #    if (i != np.inf):
    #        points.append([i,k])
    #    k += 1
        
    xPose = a.position.y * 100
    yPose = a.position.x * -100
    global i
    i += 1
    a = calAngle(xPose,yPose)
    if xPose > 0:
        a = 360 - a

    # b = calAngle(100 - xPose,100 - yPose)
    # if xPose > 10 :
    #     b = 360 - b

    b = calAngle( -1 * xPose, -1 * yPose)
    if xPose > 0 :
        b = 360 - b

    if abs(rotation - b) > 10 and not avoid:
        if i % 4 == 0:
            pub.publish("d")
            i = 0
    elif abs(rotation - b) < 10 and not avoid:
        if calDis(xPose,0,yPose,0) > 20:
            if i % 6 == 0:
                pub.publish("w")
                i = 0
    
    print("Robot at ", str(xPose),str(yPose))
    print("angle x --> O", str(a))
    print('rotation ', str(rotation))
    print('100,100 --> R', str(b))
    print(calDis(xPose,100,yPose,100))
    
    # dis = calDis(xPose,trajec[0][0],yPose,trajec[0][1])
    # dA = np.arcsin(abs(xPose - trajec[0][0]) / dis)

    # if rotation > dA:
    #     delta = round(rotation + dA)
    # else:
    #     delta = round(dA - rotation)
    
    # obs= 0
    # for i in points:
    #     if abs(delta - i[1]) < 5:
    #         obs = i[0]
    #         break
    # global avoid
    # if obs < dis:
    #     avoid = True
    
    # if not avoid :
    #   if dis > 10:
    #        if delta < 5:
    #            ser.write(b'f')
    #            rospy.sleep(1)
    #        else:
    #            ser.write(b'r')
    #    else:
    #        trajec.remove(0)
    #else:
    #    print("asd")

    # if dis > 30 and (not avoid):
    #     dA = np.arcsin(abs(xPose - trajec[0][0]) / dis)
    #     if trajec[0][0] > xPose and trajec[0][1] < yPose:
    #         dA = 180 - dA
    #     elif trajec[0][0] < xPose and trajec[0][1] < yPose:
    #         dA = 180 + dA
    #     elif trajec[0][0] < xPose and trajec[0][1] > yPose:
    #         dA = 360 - dA
    #     global avoid
    #     if not avoid:
    #         if abs(rotation - dA) > 15: 
    #             ser.write(b'r')
    #             rospy.sleep(1.2)
    #         else:
    #             for i in check:
    #                 if i[1] < 60:
    #                     avoid = True
    #             if not avoid: 
    #                 ser.write(b'f')
    #                 rospy.sleep(1.2)
    # else:
    #     tempPoint = [[xPose,trajec[1][1]] , [trajec[1][0],yPose]]
    #     tempDis = [calDis(xPose,i[0],yPose,i[1]) for i in tempPoint]
    #     trajec.pop(0)
    #     if tempDis[0] > tempDis[1]:
    #         trajec.insert(tempPoint[0])
    #     else:
    #         trajec.insert(tempPoint[1])

if __name__ == '__main__':

    trajec = np.loadtxt('/home/phuong/catkin_ws/src/beginner_tutorials/trajec.txt')
    rospy.init_node("plotter")
    rp = message_filters.Subscriber("scan",LaserScan)
    pose = message_filters.Subscriber("slam_out_pose", PoseStamped)
    ts = message_filters.ApproximateTimeSynchronizer([rp, pose], 10, 0.1, allow_headerless=True)

    ts.registerCallback(getdata)
    #rospy.Subscriber("slam_out_pose",PoseStamped,getdata)
    rospy.spin()