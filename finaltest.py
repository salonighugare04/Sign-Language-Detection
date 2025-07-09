import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

#Single Hand Parameters
OFFSET_SINGLE = 20
IMG_SIZE_SINGLE = 200
CONFIDENCE_THRESHOLD_SINGLE = 0.95
STABLE_SIGN_DURATION_SINGLE = 1.0
SINGLE_HAND_MODEL_PATH = "Models/keras_model.h5"
SINGLE_HAND_LABELS_PATH = "Models/labels.txt"
single_hand_labels = ["DoYou", "Hello", "ILoveYou", "No", "Listening", "ThankYou", "toTalk", "Want", "Yes", "GoodBye",
                      "Sorry", "Need", "I"]

#Two Hand Parameters
OFFSET_DOUBLE = 20
IMG_SIZE_DOUBLE = 200
CONFIDENCE_THRESHOLD_DOUBLE = 0.95  # Adjust as needed
STABLE_SIGN_DURATION_DOUBLE = 1.0
DOUBLE_HAND_MODEL_PATH = "Models1/keras_model.h5"
DOUBLE_HAND_LABELS_PATH = "Models1/labels.txt"
double_hand_labels = ["Home", "Name", "Stop", "Play", "When"]

detector = HandDetector(maxHands=2)
classifier_single = Classifier(SINGLE_HAND_MODEL_PATH, SINGLE_HAND_LABELS_PATH)
classifier_double = Classifier(DOUBLE_HAND_MODEL_PATH, DOUBLE_HAND_LABELS_PATH)

sentence = ""
current_sign = ""
last_sign = ""
last_sign_change_time = time.time()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        continue

    img_output = img.copy()
    hands, img = detector.findHands(img)

    if hands and len(hands) == 2:

        hand1 = hands[0]
        bbox1 = hand1['bbox']
        hand2 = hands[1]
        bbox2 = hand2['bbox']

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        xmin = min(x1, x2)
        ymin = min(y1, y2)
        xmax = max(x1 + w1, x2 + w2)
        ymax = max(y1 + h1, y2 + h2)
        w_combined = xmax - xmin
        h_combined = ymax - ymin

        if w_combined > 0 and h_combined > 0:
            img_white = np.ones((IMG_SIZE_DOUBLE, IMG_SIZE_DOUBLE, 3), np.uint8) * 255
            img_crop = img[ymin - OFFSET_DOUBLE:ymax + OFFSET_DOUBLE, xmin - OFFSET_DOUBLE:xmax + OFFSET_DOUBLE]

            if img_crop.size > 0:
                aspect_ratio = h_combined / w_combined

                try:
                    if aspect_ratio > 1:
                        k = IMG_SIZE_DOUBLE / h_combined
                        w_cal = math.ceil(k * w_combined)
                        img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE_DOUBLE))
                        w_gap = math.ceil((IMG_SIZE_DOUBLE - w_cal) / 2)
                        img_white[:, w_gap:w_cal + w_gap] = img_resize
                    else:
                        k = IMG_SIZE_DOUBLE / w_combined
                        h_cal = math.ceil(k * h_combined)
                        img_resize = cv2.resize(img_crop, (IMG_SIZE_DOUBLE, h_cal))
                        h_gap = math.ceil((IMG_SIZE_DOUBLE - h_cal) / 2)
                        img_white[h_gap:h_cal + h_gap, :] = img_resize

                    prediction, index = classifier_double.getPrediction(img_white, draw=False)
                    confidence = prediction[index]

                    if confidence > CONFIDENCE_THRESHOLD_DOUBLE:
                        current_sign = double_hand_labels[index]
                        if current_sign != last_sign:
                            last_sign = current_sign
                            last_sign_change_time = time.time()
                        elif (time.time() - last_sign_change_time) > STABLE_SIGN_DURATION_DOUBLE:
                            if last_sign and (not sentence.endswith(last_sign + " ")):
                                sentence += last_sign + " "
                                last_sign = ""

                    # Get size of the text
                    text = f"{current_sign} ({confidence * 100:.1f}%)"
                    font = cv2.FONT_HERSHEY_COMPLEX
                    font_scale = 1
                    font_thickness = 2
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_width = text_size[0] + 20  # Add some padding

                    # Display of current sign
                    cv2.rectangle(img_output, (xmin - OFFSET_DOUBLE, ymin - OFFSET_DOUBLE - 70),
                                  (xmin - OFFSET_DOUBLE + text_width, ymin - OFFSET_DOUBLE + 60 - 50), (255, 0, 255),
                                  cv2.FILLED)
                    cv2.putText(img_output, text,
                                (xmin, ymin - 45), font, font_scale, (0, 0, 0), font_thickness)
                    cv2.rectangle(img_output, (xmin - OFFSET_DOUBLE, ymin - OFFSET_DOUBLE),
                                  (xmax + OFFSET_DOUBLE, ymax + OFFSET_DOUBLE), (255, 0, 255),
                                  4)  # Keep the combined hand bounding box
                except Exception as e:
                    print(f"Error processing two-hand image: {e}")
            else:
                current_sign = ""
                last_sign = ""
        else:
            current_sign = ""
            last_sign = ""

    elif hands and len(hands) == 1:

        hand = hands[0]
        x, y, w, h = hand['bbox']

        img_white = np.ones((IMG_SIZE_SINGLE, IMG_SIZE_SINGLE, 3), np.uint8) * 255
        img_crop = img[y - OFFSET_SINGLE:y + h + OFFSET_SINGLE, x - OFFSET_SINGLE:x + w + OFFSET_SINGLE]

        if img_crop.size > 0:
            aspect_ratio = h / w

            try:
                if aspect_ratio > 1:
                    k = IMG_SIZE_SINGLE / h
                    w_cal = math.ceil(k * w)
                    img_resize = cv2.resize(img_crop, (w_cal, IMG_SIZE_SINGLE))
                    w_gap = math.ceil((IMG_SIZE_SINGLE - w_cal) / 2)
                    img_white[:, w_gap:w_cal + w_gap] = img_resize
                else:
                    k = IMG_SIZE_SINGLE / w
                    h_cal = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (IMG_SIZE_SINGLE, h_cal))
                    h_gap = math.ceil((IMG_SIZE_SINGLE - h_cal) / 2)
                    img_white[h_gap:h_cal + h_gap, :] = img_resize

                prediction, index = classifier_single.getPrediction(img_white, draw=False)
                confidence = prediction[index]

                if confidence > CONFIDENCE_THRESHOLD_SINGLE:
                    current_sign = single_hand_labels[index]
                    if current_sign != last_sign:
                        last_sign = current_sign
                        last_sign_change_time = time.time()
                    elif (time.time() - last_sign_change_time) > STABLE_SIGN_DURATION_SINGLE:
                        if last_sign and (not sentence.endswith(last_sign + " ")):
                            sentence += last_sign + " "
                            last_sign = ""


                text = f"{current_sign} ({confidence * 100:.1f}%)"
                font = cv2.FONT_HERSHEY_COMPLEX
                font_scale = 1
                font_thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_width = text_size[0] + 20  # Add some padding

                # Display of current sign
                cv2.rectangle(img_output, (x - OFFSET_SINGLE, y - OFFSET_SINGLE - 70),
                              (x - OFFSET_SINGLE + text_width, y - OFFSET_SINGLE +60 -50), (0, 255, 0), cv2.FILLED)
                cv2.putText(img_output, text,
                            (x, y - 45), font, font_scale, (0, 0, 0), font_thickness)
                cv2.rectangle(img_output, (x - OFFSET_SINGLE, y - OFFSET_SINGLE),
                              (x + w + OFFSET_SINGLE, y + h + OFFSET_SINGLE), (0, 255, 0),
                              4)  # Keep the hand bounding box

            except Exception as e:
                print(f"Error processing single-hand image: {e}")
    else:
        current_sign = ""
        last_sign = ""

    # Display of sentence
    cv2.putText(img_output, "Sentence:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(img_output, sentence, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Sign Language Recognition', img_output)

    key = cv2.waitKey(1)
    if key == ord('x'):
        break
    if key == ord('c'):
        sentence = ""
        current_sign = ""
        last_sign = ""

cap.release()
cv2.destroyAllWindows()
