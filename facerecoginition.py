import cv2
import numpy as np
import face_recognition
import os

imgmichal = face_recognition.load_image_file('model images/michal.jpeg')
imgmichal = cv2.cvtColor(imgmichal,cv2.COLOR_BGR2RGB)
imgmouli = face_recognition.load_image_file('model images/mouli.jpeg')
imgmouli = cv2.cvtColor(imgmouli,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgmichal)[0]
encodemichal = face_recognition.face_encodings(imgmichal)[0]
cv2.rectangle(imgmichal,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

facemouli = face_recognition.face_locations(imgmouli)[0]
encodemouli = face_recognition.face_encodings(imgmouli)[0]
cv2.rectangle(imgmouli,(facemouli[3],facemouli[0]),(facemouli[1],facemouli[2]),(255,0,255),2)

result = face_recognition.compare_faces([encodemichal],encodemouli)
facedis = face_recognition.face_distance([encodemichal],encodemouli)
print(result,facedis)

cv2.imshow('michal',imgmichal)
cv2.imshow('mouli',imgmouli)

cv2.waitKey(0)