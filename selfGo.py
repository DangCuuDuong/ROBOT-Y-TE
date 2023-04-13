#!/usr/bin/env python2
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
import math
from geometry_msgs.msg import PoseStamped
import message_filters
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import serial
ser = serial.Serial('/dev/ttyUSB1', 115200)
x = []
y = []
trajec = []
counter = 0
avoid = False

def calDis(x1,x2,y1,y2):
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

def getdata(rp,pose):
    
    if not trajec:
        rospy.signal_shutdown("dont know where to go")

    a = pose.pose
    rot = [a.orientation.x,a.orientation.y,a.orientation.z,a.orientation.w]
    (roll, pitch, yaw) = euler_from_quaternion (rot)

    rotation =math.degrees(yaw)
    if rotation < 0:
        rotation = rotation + 360
    sensor = np.array(rp.ranges) * 100
    k = 0
    points = []
    for i in sensor:
        if (i != np.inf):
            points.append([i,k])
        k += 1
        
    xPose = a.position.y * 100
    yPose = a.position.x * -100
    dis = calDis(xPose,trajec[0][0],yPose,trajec[0][1])
    dA = np.arcsin(abs(xPose - trajec[0][0]) / dis)

    if rotation > dA:
        delta = round(rotation + dA)
    else:
        delta = round(dA - rotation)
    
    obs= 0
    for i in points:
        if abs(delta - i[1]) < 5:
            obs = i[0]
            break
    global avoid
    if obs < dis:
        avoid = True
    
    if not avoid :
        if dis > 10:
            if delta < 5:
                ser.write(b'f')
                rospy.sleep(1)
            else:
                ser.write(b'r')
        else:
            trajec.remove(0)
    else:
        print("asd")

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
    rospy.init_node("plotter")
    trajec = np.loadtxt('trajec.txt')
    rp = message_filters.Subscriber("scan",LaserScan)
    pose = message_filters.Subscriber("slam_out_pose", PoseStamped)
    ts = message_filters.TimeSynchronizer([rp,pose],100)
    ts.registerCallback(getdata)
    rospy.spin()



    