import cv2,time
import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import os

#load model
model = model_from_json(open("/Users/hojinlee/Documents/Codes/ComputerVision/model.json", "r").read())
#load weights
model.load_weights('/Users/hojinlee/Documents/Codes/ComputerVision/model.h5')

# video capture
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("/Users/hojinlee/Documents/Codes/ComputerVision/haarcascade_frontalface_default.xml")

a = 1 
while True:

    a += 1
    check, frame = video.read()

    if frame is not None: 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
            roi_gray = gray[y:y+w,x:x+h] #cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            predictions = model.predict(img_pixels)

            #find max indexed array
            max_index = np.argmax(predictions[0])
            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            predicted_emotion = emotions[max_index]

            cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        resized_img = cv2.resize(frame, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    else:
        print("empty frame")
        exit(1)
    

print(a)
video.release()
cv2.destroyAllWindows()