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
        #mask = cv2.imread('./mask_1.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH))
        )
        for (x,y,w,h) in faces:
           # dim = (w, h)
           # mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
           # maskgray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
           # roi = img[x-40:x+w-40, y+40:y+h+40]
           # ret, frame_m = cv2.threshold(maskgray, 10, 255, cv2.THRESH_BINARY)
           # mask_inv = cv2.bitwise_not(frame_m)
           # img_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
           # mask_fg = cv2.bitwise_and(mask,mask,mask = frame_m)
           # dst = cv2.add(img_bg,mask_fg)
           # img[x-40:x+w-40, y+40:y+h+40] = dst
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('cam', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    print('\n [INFO] Exit\n')
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    open_camera()
