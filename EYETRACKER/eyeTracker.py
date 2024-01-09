import cv2
import ObjectDetectionModule as odm

import SerialModule as sm
import numpy as np
import time

framewidth = 640
frameHeight =480

cap = cv2.VideoCapture(0)

ser = sm.initConnection('/dev/ttyACM0',9600)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

perrorLR, perrorUD = 0,0

def findCenter(imgObjects,objects):
    cx,cy = -1,-1
    if len(objects)!=0:
        x,y,w,h = objects[0][0]
        cx = x + w//2
        cy = y + h//2
        cv2.circle(imgObjects,(cx,cy),2,(0,255,0),cv2.FILLED)
        ih, iw, ic = imgObjects.shape
        cv2.line(imgObjects,(iw//2,cy),(cx,cy),(0,255,0),1)
        cv2.line(imgObjects,(cx,ih//2),(cx,cy),(0,255,0),1)
    return cx,cy,imgObjects


def trackObject(cx,cy,w,h):
    global perrorLR, perrorUD
    kLR = [0.5,0.5]
    kUD = [0.5,0.5]

    
    if cx!=-1:
        #LEFT AND RIGHT
        errorLR = w//2 -cx
        posX = kLR[0]* errorLR + kLR[1]* (errorLR-perrorLR) #PID CONTROLLER
        posX = int(np.interp(posX,[-w//2,w//2],[90-45,90+45]))
        perrorLR = errorLR

        #UP AND DOWN
        errorUD = h//2 -cy
        posY = kUD[0] *errorUD + kUD[1] *(errorUD-perrorUD) #PID CONTROLLER
        posY = int(np.interp(posY,[-h//2,h//2],[90-45,90+45]))
        perrorUD = errorUD

        sm.sendData(ser, [posX,posY], 3)



while True:
    success, img = cap.read()
    img = cv2.resize(img, (0,0), None, 0.3,0.3)
    imgObjects, objects = odm.findObjects(img,faceCascade,1.1,7)
    cx, cy , imgObjects = findCenter(imgObjects, objects)

    h,w,c = imgObjects.shape
    cv2.line(imgObjects,(w//2,0),(w//2,h),(255,0,255),1)
    cv2.line(imgObjects,(0,h//2),(w,h//2),(255,0,255),1)

    trackObject(cx, cy, w, h)

    img = cv2.resize(imgObjects, (0,0), None, 3,3)
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord("s"):
        #sm.sendData(ser, [90,90], 3)
        break