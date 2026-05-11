"""This module captures hand gesture images using a webcam and saves them for training a deep learning model.""" 

import math
import string
import time

import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0) # 0 is the id number
detector = HandDetector(maxHands=1)
classifier = Classifier("data/model.keras", "data/label.txt")

offset = 20
img_size = 300
counter = 0
labels = list(string.ascii_uppercase) + [str(i) for i in range(10)]

# We save images when we press the S key (to train our model for example)
folder = "data/I"


def launch_webcam() -> None :
    global counter
    while True:
        success, img = cap.read()
        img_output = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Reference white image
            img_white = np.ones((img_size, img_size, 3), np.uint8) * 255

            # Check the limits
            img_height, img_width = img.shape[:2]
            y1 = max(0, y - offset)
            y2 = min(img_height, y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img_width, x + w + offset)
            
            if y2 > y1 and x2 > x1:
                img_crop = img[y1:y2, x1:x2]
                
                # Compute the aspect ratio
                aspect_ratio = h / w
                
                if aspect_ratio > 1: 
                    k = img_size / h
                    w_calculated = math.ceil(k * w)
                    img_resize = cv2.resize(img_crop, (w_calculated, img_size))
                    
                    # Center the cropped image horizontally
                    w_gap = math.ceil((img_size - w_calculated) / 2)
                    img_white[:, w_gap:w_gap + w_calculated] = img_resize
                    prediction, index = classifier.getPrediction(img_white, draw=False)
                    
                else:
                    k = img_size / w
                    h_calculated = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (img_size, h_calculated))
                    
                    # Center the cropped image vertically
                    h_gap = math.ceil((img_size - h_calculated) / 2)
                    img_white[h_gap:h_gap + h_calculated, :] = img_resize
                    prediction, index = classifier.getPrediction(img_white, draw=False)
                
                cv2.imshow("Image_White", img_white)

            cv2.putText(img_output, labels[index], (x,y-offset), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,255), 2)
            cv2.rectangle(img_output, (x-offset,y-offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
            cv2.imshow("Image_White", img_white)

            # Press the S key to save the image
            key = cv2.waitKey(1)
            if key == ord("s"):
                counter += 1
                cv2.imwrite(f'{folder}/image_{time.time()}.jpg', img_white)
                print(counter)

        else:
            # No hands
            cv2.imshow("Image", img_output)
            if cv2.waitKey(1) == ord("q"):
                break

        # Get the whole image captured by the camera
        cv2.imshow("Image", img_output)
        if cv2.waitKey(1) == ord("q"):
            break