import numpy as np

from model import *
import cv2


model = get_model()
model.load_weights(saved_weights_name)


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (300, 300))
    width, height, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96))
    predictions = model.predict(gray.reshape(-1, 96, 96, 1))
    cv2.circle(frame, (int(width * (predictions[0][0] / 96)), int(height * (predictions[0][1] / 96))), 4, (0, 0, 255),
               3, cv2.LINE_AA)
    cv2.circle(gray, (predictions[0][0], predictions[0][1]), 4, (0, 0, 255),
               3, cv2.LINE_AA)
    cv2.putText(frame, "Press 'q' to Quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Stream', frame)
    cv2.imshow('gray', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
