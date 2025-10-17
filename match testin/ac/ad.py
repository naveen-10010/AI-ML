import cv2
import face_recognition
#import serial
import time
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

known_faces_dir = "admin"
known_faces_encodings = []

for filename in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, filename)
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        known_faces_encodings.append(face_encodings[0])

frame_width = 400
frame_height = 400

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)



while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_image = frame[y:y + h, x:x + w]
        cv2.imwrite('captured_face.jpg', face_image)

        captured_image = face_recognition.load_image_file("captured_face.jpg")
        captured_face_encodings = face_recognition.face_encodings(captured_image)

        if len(captured_face_encodings) > 0:
            captured_face_encoding = captured_face_encodings[0]
            results = face_recognition.compare_faces(known_faces_encodings, captured_face_encoding)
            if any(results):
                print("Match found!")
               
            else:
                print("No match found.")
                
        else:
            print("No face detected in the captured image.")

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
