#!/usr/bin/python3

import cv2 as cv2


face_cascade = cv2\
    .CascadeClassifier('../assets/haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)


def open_camera():
    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH))
        )
        for (x,y,w,h) in faces:
            kernel_w = (w // 4) | 1
            kernel_h = (h // 4) | 1
            roi = img[x:x+w, y:y+h]
            roi = cv2.GaussianBlur(roi, (kernel_w, kernel_h), 0)
            img[x:x+w, y:y+h] = roi
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.imshow('cam', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    print('\n [INFO] Exit\n')
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    open_camera()
