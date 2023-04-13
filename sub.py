#!/usr/bin/env python3
from scipy import sparse
import rospy
import time
from gtts import gTTS
from std_msgs.msg import String
import numpy as np
import sensor_msgs.msg._Image
import cv2
import itertools
import serial
from keras_vggface.vggface import VGGFace 
import os
from scipy.spatial.distance import cosine
from keras_vggface.utils import preprocess_input
from PIL import Image, ImageTk
from pyzbar.pyzbar import decode
from pyzbar import pyzbar
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
img = 0
fnumbs = 0
import os
import playsound
threshold = 0.15
cascadePath = "/home/phuong/catkin_ws/src/beginner_tutorials/scripts/haarcascade_frontalface_default.xml"
facecascade = cv2.CascadeClassifier(cascadePath)
first = ""
last = ""
check = False
cSer = [0,0]
ser = serial.Serial('/dev/ttyUSB2',115200)
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
pub = rospy.Publisher("move",String,queue_size=10)
minW = 64
minH = 48
model = VGGFace(model = "resnet50",include_top = False, input_shape = (224,224,3))
data_face ,persons = get_dataset()
pred_dataset = get_embedding(data_face,model)

def speech(text):
    tts = gTTS(text=text, lang='vi', slow=False)
    tts.save("sound.mp3")
    playsound.playsound("sound.mp3")
    os.remove("sound.mp3")

def postprocess(frame, outs):
    frameHeight, frameWidth = frame.shape[:2]

    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold:
                x, y, width, height = detection[:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                left = int(x - width / 2)
                top = int(y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, int(width), int(height)])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, threshold - 0.1)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cropped_image = frame[top:top + height, left:left + width]

        # Draw bounding box for objects
        
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), 1)

        # Draw class name and confidence
        # label = '%s:%.2f' % (classes[classIds[i]], confidences[i])
        # cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

def callback(data):

    # img = bridge.imgmsg_to_cv2(data)
    img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    global check
    if not check:
        faces = facecascade.detectMultiScale(img,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW),int(minH)),)
        id = "unknown"
        speak = []           
        if cSer[1] == 0:
            if cSer[0] < 90:
                cSer[0] += 1
                ser.write(b'u')
            else:
                cSer[1] = 1
        elif cSer[1] == 1:
            if cSer[0] > 0:
                cSer[0] -= 1
                ser.write(b'd')
            else:
                cSer[1] = 0
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
        if "BaoLong" in list(itertools.chain.from_iterable(speak)) or "CuuDuong" in list(itertools.chain.from_iterable(speak)):
            for p in speak:
                if p[0] == "CuuDuong" or p[0] == "BaoLong" or p[0] == "CuuDuong2" or p[0] == "BaoLong2":
                    check = True
                    if p[0] == "CuuDuong" or p[0] == "CuuDuong2":
                        speech("xin chào bác sĩ Cửu Dương")
                        break
                    if p[0] == "BaoLong" or p[0] == "BaoLong2":
                        playsound.playsound('/home/phuong/benhnhan.mp3')
                        break

    else:        
    #imgHeight, imgWidth = img.shape[:2]
    #blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    #ln = net.getLayerNames()
    #ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    #net.setInput(blob)
    #outs = net.forward(ln)
    # postprocess(img, outs)
        
        global fnumbs
        fnumbs += 1
        detectedBarcodes = pyzbar.decode(img)
        # codes, img = reader.extract(img, True)
        if detectedBarcodes:
            (x, y, w, h) = detectedBarcodes[0].rect
            b = y + h/2
            if b < 360 :
                ser.write(b'u')
            if b > 720 :
                ser.write(b'd')
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)
            if fnumbs % 3 == 0:
                a = x + w/2
                if a < 1160 and a > 768:
                    if w > 100 :
                        pub.publish("w")
                if a <= 768:
                    pub.publish("a")
                if a >= 1160:
                    pub.publish("d")
                fnumbs = 0 
                
            
    cv2.imshow('',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        rospy.signal_shutdown("shutdown")

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/kinect2/hd/image_color", sensor_msgs.msg.Image, callback)
    # rospy.Subscriber("/camera/rgb/image_color", sensor_msgs.msg.Image, callback)
    rospy.spin()

if __name__ == '__main__':
    #classes = open('/home/phuong/catkin_ws/src/beginner_tutorials/scripts/qrcode.names').read().strip().split('\n')
    #net = cv2.dnn.readNetFromDarknet('/home/phuong/catkin_ws/src/beginner_tutorials/scripts/qrcode-yolov3-tiny.cfg', '/home/phuong/catkin_ws/src/beginner_tutorials/scripts/qrcode-yolov3-tiny.weights')
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    listener()
    
