import math
import time

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0) # 0 is the id number
detector = HandDetector(maxHands=1)

offset = 20
img_size = 300
counter = 0

# We save images when we press the S key (to train our model)
folder = "data/A"

while True:
    success, img = cap.read()
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
                
            else:
                k = img_size / w
                h_calculated = math.ceil(k * h)
                img_resize = cv2.resize(img_crop, (img_size, h_calculated))
                
                # Center the cropped image vertically
                h_gap = math.ceil((img_size - h_calculated) / 2)
                img_white[h_gap:h_gap + h_calculated, :] = img_resize
            
            cv2.imshow("Image_White", img_white)

        cv2.imshow("Image_White", img_white)

    # Get the whole image captured by the camera
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg', img_white)
        print(counter)