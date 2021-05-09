#!/usr/bin/python3

import cv2 as cv2
import numpy as np
import uuid
import json


face_cascade = cv2\
    .CascadeClassifier('./assets/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

username = input('\n [INPUT] Enter username:\t')

with open('./config/users.json', 'r') as f:
    data = json.load(f)
with open('./config/users.json', 'w') as f:
    file_data = data
    file_data['names'].append(username)
    json.dump(file_data, f, indent = 4)
print(f' [INFO] {username} username appended to json\n')

face_id = len(data['names'])
face_uuid = str(uuid.uuid4())

def open_camera():
    count = 0
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            count += 1
            file_name = f'./dataset/{face_uuid}_{face_id}_{count}.png'
            cv2.imwrite(file_name, gray[y:y+h,x:x+w])
            print(f' [INFO] Saved into {file_name}')
        cv2.imshow('cam', img)
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >= 30:
            break
    print('\n [INFO] Exit \n')
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    open_camera()
