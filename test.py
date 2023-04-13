import numpy as np
import cv2
import face_recognition
from skimage.measure import compare_ssim
from mtcnn_cv2 import MTCNN
from sklearn.neighbors import NearestNeighbors
ksize = (101, 101)
font = cv2.FONT_HERSHEY_SIMPLEX
neigh = NearestNeighbors(n_neighbors=1)

detector = MTCNN()
img = cv2.imread('dataset/cropped.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('dataset/cropped2.jpg',cv2.IMREAD_GRAYSCALE)
neigh.fit(img,img2);

img3 = cv2.imread('dataset/User.BaoLong.jpg')
result = detector.detect_faces(img3)

cropped_image = []
if len(result) > 0:
    bounding_box = result[0]['box']
    left = bounding_box[0]
    top = bounding_box[1]
    width = bounding_box[2]
    height = bounding_box[3]
    cropped_image = img3[top:top + height, left:left + width]
    cv2.rectangle(img,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                (0,155,255),
                2)
cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
print(neigh.kneighbors(cropped_image))