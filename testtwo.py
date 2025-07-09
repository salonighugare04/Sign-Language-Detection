import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

OFFSET = 20
IMG_SIZE = 200
CONFIDENCE_THRESHOLD = 0.95  # Only accept predictions with confidence > 95%
STABLE_SIGN_DURATION = 1.0  # Seconds a sign must be stable before adding to sentence

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Models1/keras_model.h5 ", "Models1/labels.txt")      #put the model file path of your own

labels = ["DoYou", "Hello", "ILoveYou", "No", "Listening", "Please", "ThankYou", "toTalk", "Want", "Yes"]  


sentence = ""
current_sign = ""
last_sign = ""
last_sign_change_time = time.time()
last_sign_confidence = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    img_output = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
        img_crop = img[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]

        if img_crop.size > 0:
            aspect_ratio = h / w

            try:
                if aspect_ratio > 1:
                    k = IMG_SIZE / h
                    w_cal = math.ceil(k * w)
                    img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE))
                    w_gap = math.ceil((IMG_SIZE - w_cal) / 2)
                    img_white[:, w_gap:w_cal + w_gap] = img_resize
                else:
                    k = IMG_SIZE / w
                    h_cal = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (IMG_SIZE, h_cal))
                    h_gap = math.ceil((IMG_SIZE - h_cal) / 2)
                    img_white[h_gap:h_cal + h_gap, :] = img_resize

                prediction, index = classifier.getPrediction(img_white, draw=False)
                confidence = prediction[index]

                if confidence > CONFIDENCE_THRESHOLD:
                    current_sign = labels[index]

                    # Check if sign has changed
                    if current_sign != last_sign:
                        last_sign = current_sign
                        last_sign_change_time = time.time()
                        last_sign_confidence = confidence
                    else:
                        # Check if sign has been stable for required duration
                        if (time.time() - last_sign_change_time) > STABLE_SIGN_DURATION:
                            if last_sign and (not sentence.endswith(last_sign + " ")):
                                sentence += last_sign + " "
                                last_sign = ""  # Reset to avoid duplicates

                # Display current sign (even if not added to sentence yet)
                cv2.rectangle(img_output, (x - OFFSET, y - OFFSET - 70),
                              (x - OFFSET + 400, y - OFFSET + 60 - 50), (0, 255, 0), cv2.FILLED)
                cv2.putText(img_output, f"{current_sign} ({confidence * 100:.1f}%)",
                            (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(img_output, (x - OFFSET, y - OFFSET),
                              (x + w + OFFSET, y + h + OFFSET), (0, 255, 0), 4)

            except Exception as e:
                print(f"Error processing image: {e}")
    else:
        current_sign = ""
        last_sign = ""

    # Display sentence
    cv2.putText(img_output, "Sentence:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(img_output, sentence, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # # Display instructions
    # cv2.putText(img_output, "Press 'c' to clear, 'x' to exit", (10, img.shape[0] - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Sign Language Recognition', img_output)

    key = cv2.waitKey(1)
    if key == ord('x'):    #Press X to terminate
        break
    if key == ord('c'):    #Press C to clear the text
        sentence = ""
        current_sign = ""
        last_sign = ""

cap.release()
cv2.destroyAllWindows


