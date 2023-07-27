# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 15:06:23 2021

@author: nguyen
"""
import cv2
import numpy as np


cap = cv2.VideoCapture("1N4148WS_2.avi")

# Check if camera opened successfully
if (cap.isOpened()== False):
    
    print("Error opening video stream or file")

	 

# Objet detection from stable camera

object_detector = cv2.createBackgroundSubtractorMOG2(history = 400 , varThreshold = 10)

while (cap.isOpened()):
    ret, frame = cap.read()
    
    #List of contours
    matched_contours = []
    
    if ret == True:
        
        #extract roi 
        
        roi = frame[0:600,500:1280]
        height,width,_ = frame.shape
        #print(height,width)
        
        #roi = frame[400:2500,500:2500]
        # Object detection
        mask = object_detector.apply(roi)
        
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            
            #Calculate area and remove small elements
            
            area = cv2.contourArea(cnt)
            
            if area > 600:
                #cv2.drawContours(roi, [cnt], -1, (0,255,0),2)
                x , y , w , h = cv2.boundingRect(cnt)
                cv2.rectangle(roi, (x,y), (x + w , y + h) , (0,255,0),3)
        
        
        cv2.imshow("roi", roi)
        cv2.imshow("Frame",frame)
        cv2.imshow("mask",mask)
        
        # Escape
        key = cv2.waitKey(30)
        if key == 27 :
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()