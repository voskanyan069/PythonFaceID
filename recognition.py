#!/usr/bin/python3

import cv2 as cv2
import os
import json


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer/trainer.yml')
cascade_path = './assets/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path);
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
with open('./config/users.json') as f:
    data = json.load(f)
    allowed_users = data['allowed_users']
names = data['names']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)


def recognize_face():
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if confidence <= 100:
                id = names[id - 2]
                if (100 - confidence) > 80:
                    if id in allowed_users:
                        print(f' [INFO] Allowed user: {id}')
            else:
                id = 'unknown'
            confidence = f'{round(100 - confidence)}%'
            cv2.putText(
                img,
                str(id),
                (x+5,y-5),
                font,
                1,
                (255,255,255),
                2
            )
            cv2.putText(
                img,
                str(confidence),
                (x+5,y+h-5),
                font,
                1,
                (255,255,255),
                1
            )
        cv2.imshow('cam', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    print('\n [INFO] Exit \n')
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
	recognize_face()
