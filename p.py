#!/usr/bin/env python3
import rospy
import numpy as np
import sensor_msgs.msg._Image
import cv2
import serial
# from cv_bridge import CvBridge
# from cv_bridge.boost.cv_bridge_boost import getCvType
from PIL import Image, ImageTk
import face_recognition
import subprocess
from keras_vggface.vggface import VGGFace 
import os
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
## SERIAL /dev/ttyUSB2 neu sai 1-2 ##
ser = serial.Serial('/dev/ttyUSB1',115200)


cascadePath = "/home/phuong/catkin_ws/src/beginner_tutorials/scripts/haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(cascadePath)
first = ""
last = ""

def preprocess_face(face,required_size =(224,224)):

    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)

    return face_array

def get_dataset(path = "/home/phuong/catkin_ws/src/beginner_tutorials/scripts/image/"):
   faces = []
   persons = []
   for name_img in os.listdir(path):
      person = name_img.split(".")[1]
      persons.append(person)
      face = cv2.imread(os.path.join(path,name_img))
      face_all = facecascade.detectMultiScale(face, 1.3, 5,minSize = (64,48))

      for (x,y,w,h) in face_all:

         faces.append(face[y:y+h,x:x+w])

   print(len(faces),len(persons))
   return faces ,persons


def get_embedding(bboxfaces,model):
    faces = [preprocess_face(f) for f in bboxfaces]

    samples = np.asarray(faces,"float32")

    samples = preprocess_input(samples,version=2)    
    
    pred = model.predict(samples)

    return pred

font = cv2.FONT_HERSHEY_SIMPLEX

minW = 64
minH = 48
model = VGGFace(model = "resnet50",include_top = False, input_shape = (224,224,3))
data_face ,persons = get_dataset()
pred_dataset = get_embedding(data_face,model)

# def execute_unix(inputcommand):
#    p = subprocess.Popen(inputcommand, stdout=subprocess.PIPE, shell=True)
#    (output, err) = p.communicate()
#    return output


def callback(data):
   
      
   # img = bridge.imgmsg_to_cv2(data)
   img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
   faces = facecascade.detectMultiScale( 
                        img,
                        scaleFactor = 1.2,
                        minNeighbors = 5,
                        minSize = (int(minW),int(minH)),  
                    )
   id = "unknown"
   speak = []           

   # face recognition 
   for (x,y,w,h) in faces:
      face = img[y:y+h,x:x+w] # 1
      cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) # 1
      # face = preprocess_face(face)
      pred = get_embedding([face],model)

      for index ,emb in enumerate(pred_dataset[:]):
            # print(len(pred[:-1]))
            score = cosine(emb,pred[0])
            if (score < 0.4):
               id = persons[index]
            speak.append([id,x,y,w,h])
            
      cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
      global first

      if speak:
         for i in speak:
            if i[0] != "unknown":
               a = i[1] + i[3]/2.0
               b = i[2] + i[4]/2.0
               if a <= 215:
                  ser.write(b'l')
               elif a >= 430:
                  ser.write(b'r')
               if b <= 160:
                  ser.write(b'u')
               elif b >= 320:
                  ser.write(b'd')
               break

    

   cv2.imshow('',img)
   if cv2.waitKey(1) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      rospy.signal_shutdown("hello")

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/kinect2/hd/image_color", sensor_msgs.msg.Image, callback)
   #  rospy.Subscriber("/camera/rgb/image_color", sensor_msgs.msg.Image, callback)
    rospy.spin()


if __name__ == '__main__':
    listener()