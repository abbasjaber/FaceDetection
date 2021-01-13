import cv2
from random import randrange

trained_faces_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# webcam = cv2.VideoCapture(0)
webcam = cv2.VideoCapture('faces.mp4')

while True:
    successful_frame_read , frame = webcam.read()
    graysacled_image = cv2.cvtColor(frame , cv2.COLOR_RGB2GRAY)
    face_coordinates = trained_faces_data.detectMultiScale(graysacled_image)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame,(x, y), (x+w,y+h),randrange(255),1)

    cv2.imshow('Waw', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()
