import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_russian_plate_number.xml')

cap = cv2.VideoCapture('CarNumberPlates1.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 250)

if(cap.isOpened() == False):
    print('ERROR VIDEO READING')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(5, 5))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        crop_NumberPlate = frame[y:y+h, x:x+w]
        blur_CNP = cv2.blur(crop_NumberPlate, ksize=(25, 25))
        frame[y:y+h, x:x+w] = blur_CNP

    if ret == True:
        cv2.imshow('Video', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
