#!/usr/bin/python3

import cv2
import numpy as np
from PIL import Image
import os


path = './dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('./assets/haarcascade_frontalface_default.xml')


def get_images_and_labels(path):
    image_paths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for image_path in image_paths:
        PIL_img = Image.open(image_path).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        
        id = int(os.path.split(image_path)[-1].split('_')[1].split('.png')[0])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids


def train():
    print ('\n [INFO] Training faces. It will take a few seconds...')
    faces, ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('./trainer/trainer.yml')
    print(f' [INFO] {len(np.unique(ids))} faces trained\n')


if __name__ == '__main__':
    train()
