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
import playsound

end = False
def speech():
    playsound.playsound("/home/phuong/catkin_ws/src/beginner_tutorials/scripts/thongbao.mp3")

pub = Publisher('move',String,queue_size=10)
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
    global i
    global avoid
    global listDis
    global end
    global rlistD
    if listDis:
        xDis = listDis[0][0]
        yDis = listDis[0][1]
        a = pose.pose
        rot = [a.orientation.x,a.orientation.y,a.orientation.z,a.orientation.w]
        (_, _, yaw) = euler_from_quaternion(rot)
        rotation =math.degrees(yaw)

        if rotation < 0:
            rotation = rotation + 360
        rotation = 360 - rotation

        xPose = a.position.y * 100
        yPose = a.position.x * -100
        
        i += 1

        b = calAngle(xDis - xPose,yDis - yPose)
        if xPose > xDis :
            b = 360 - b

        if (abs(rotation - b) > 10 and abs(rotation - b) < 350) and not avoid:
            if rotation < 180:
                if b > rotation and b < (rotation + 180) :
                    if i % 4 == 0:
                        pub.publish("d")
                        i = 0        
                else:
                    if i % 4 == 0:
                        pub.publish("a")
                        i = 0
            if rotation > 180:
                if b < rotation and b > (rotation - 180) :
                    if i % 4 == 0:
                        pub.publish("a")
                        i = 0
                else:
                    if i % 4 == 0:
                        pub.publish("d")
                        i = 0
        elif (abs(rotation - b) < 10 or abs(rotation - b) >= 350) and not avoid:
            if calDis(xPose,xDis,yPose,yDis) > 16:
                # if rp.ranges[0] > 50 or rp.ranges[3] > 50:
                if i % 6 == 0:
                    pub.publish("w")
                    i = 0
                # else:
                #     avoid = True
            else:
                if not end:
                    command = raw_input("Da di den vi tri x = %f y = %f"%(xPose,yPose) + ". Moi ban nhap lenh:\n(1). Thong bao\n(2). Di tiep\n...................................................\n")
                    if int(command) == 1:
                        speech()    
                listDis.pop(0)
        print(b)
        print(rotation)
    else:
        if not end:
            listDis = rlistD[:]
            print(listDis)
            end = True
        else:
            rospy.signal_shutdown("turn off")
    
    
    

if __name__ == '__main__':
    listDis = []
    while True:
        xTemp = int(input("nhap x(cm): "))
        yTemp = int(input("nhap y(cm): "))
        listDis.append([xTemp,yTemp])
        p = str(raw_input("wanna stop? (y) or (n): "))
        if p == "y":
            break
    rlistD = listDis[::-1]
    rlistD.pop(0)
    rlistD.append([0,0])
    print(listDis)
    print(rlistD)
    # trajec = np.loadtxt('/home/phuong/catkin_ws/src/beginner_tutorials/trajec.txt')
    rospy.init_node("pkjnlotter")
    rp = message_filters.Subscriber("scan",LaserScan)
    pose = message_filters.Subscriber("slam_out_pose", PoseStamped)
    ts = message_filters.ApproximateTimeSynchronizer([rp, pose], 10, 0.1, allow_headerless=True)

    ts.registerCallback(getdata)
    #rospy.Subscriber("slam_out_pose",PoseStamped,getdata)
    rospy.spin()