import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

# Initialize video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Initialize detector and classifier
detector = HandDetector(maxHands=1)

# Use relative path or environment variable for model files
model_path = os.path.join("Models", "keras_model.h5")
labels_path = os.path.join("Models", "labels.txt")

if not os.path.exists(model_path) or not os.path.exists(labels_path):
    print("Error: Model files not found")
    exit()

classifier = Classifier(model_path, labels_path)

# Configuration
offset = 20
imgSize = 200
labels = ["DoYou", "Hello", "ILoveYou", "No", "Listening", "Please", "ThankYou", "toTalk", "Want", "Yes"]
displayed_signs = []  # List to track signs displayed on the frame
display_duration = 2  # Duration to display each sign (in seconds)
start_time = 0

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Skip processing if crop area is invalid
        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        try:
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

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            detected_sign = labels[index]
            confidence = prediction[index]

            # Display detected sign on the image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                          (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, f"{detected_sign} ({confidence:.2f})",
                        (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (0, 255, 0), 4)

            # Control sign display duration
            current_time = cv2.getTickCount() / cv2.getTickFrequency()  # Get time in seconds
            if detected_sign not in displayed_signs:
                displayed_signs.append(detected_sign)
                start_time = current_time

            if current_time - start_time > display_duration:
                displayed_signs.clear()  # Clear displayed signs after duration
                displayed_signs.append(detected_sign)
                start_time = current_time

            # Display the currently displayed signs below the video
            display_text = "Detected: " + ", ".join(displayed_signs)
            cv2.putText(imgOutput, display_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Optional: Show processing windows
            # cv2.imshow('ImageCrop', imgCrop)
            # cv2.imshow('ImageWhite', imgWhite)

        except Exception as e:
            print(f"Processing error: {e}")
            continue

    # Display main output
    cv2.imshow('Sign Language Recognition', imgOutput)

    # Exit on 'x' key
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()