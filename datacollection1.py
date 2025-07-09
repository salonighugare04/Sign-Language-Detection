import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = r"C:\Users\salon\PycharmProjects\MiniSign\Data\Need"    # Path of folder to save collected data

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    # Only proceed if a hand is detected
    if hands:
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
                if key == ord("s"):     # Press key S to save the image
                    counter += 1
                    cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                    print(f"Image {counter} saved.")

    cv2.imshow('Image', img)

    # To exit press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
