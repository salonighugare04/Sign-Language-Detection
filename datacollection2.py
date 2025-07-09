import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Set maxHands to 2
offset = 20
imgSize = 300
counter = 0

folder = r"C:\Users\salon\PycharmProjects\MiniSign\Data\When"  # Modified folder name
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    # Only proceed if at least one hand is detected
    if hands:
        # Process both hands if detected
        if len(hands) == 2:
            hand1 = hands[0]
            bbox1 = hand1['bbox']
            hand2 = hands[1]
            bbox2 = hand2['bbox']

            # Find the combined bounding box of both hands
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2

            xmin = min(x1, x2)
            ymin = min(y1, y2)
            xmax = max(x1 + w1, x2 + w2)
            ymax = max(y1 + h1, y2 + h2)
            w_combined = xmax - xmin
            h_combined = ymax - ymin

            if w_combined > 0 and h_combined > 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[ymin - offset:ymax + offset, xmin - offset:xmax + offset]

                if imgCrop.size > 0:
                    imgCropShape = imgCrop.shape
                    aspectRatio = h_combined / w_combined

                    if aspectRatio > 1:
                        k = imgSize / h_combined
                        wCal = math.ceil(k * w_combined)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w_combined
                        hCal = math.ceil(k * h_combined)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    cv2.imshow('ImageCrop', imgCrop)
                    cv2.imshow('ImageWhite', imgWhite)

                    key = cv2.waitKey(1)
                    if key == ord("s"):  # Press key S to save the image
                        counter += 1
                        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                        print(f"Image {counter} saved.")
        elif len(hands) == 1:
            # You might want to handle the case where only one hand is detected differently
            hand = hands[0]
            x, y, w, h = hand['bbox']

            if w > 0 and h > 0:
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                if imgCrop.size > 0:
                    imgCropShape = imgCrop.shape
                    aspectRatio = h / w

                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    cv2.imshow('ImageCrop', imgCrop)
                    cv2.imshow('ImageWhite', imgWhite)

                    key = cv2.waitKey(1)
                    if key == ord("s"):  # Press key S to save the image
                        counter += 1
                        cv2.imwrite(f'{folder}/Image_single_{time.time()}.jpg', imgWhite)
                        print(f"Single hand image {counter} saved.")

    cv2.imshow('Image', img)

    # To exit press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()