import tkinter as tk

def execute_function():
  import cv2
  import face_recognition
  import serial
  import time
  import os

  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  known_faces_dir = "known_faces"
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

  # Establish serial communication with Arduino
  ser = serial.Serial('COM4', 9600)  # Replace 'COM3' with the appropriate serial port
  t=0
  while t==0:
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
              t=1
              captured_face_encoding = captured_face_encodings[0]
              results = face_recognition.compare_faces(known_faces_encodings, captured_face_encoding)
              if any(results):
                  print("Match found!")
                  #ser.write(b'1')  # Send '1' to Arduino via serial
                  
              else:
                  print("No match found.")
                  #ser.write(b'0')  # Send '0' to Arduino via serial
          
          else:
              print("No face detected in the captured image.")
              
      #ser.close()
      cv2.imshow('Face Detection', frame)

      

  cap.release()
  cv2.destroyAllWindows()

def execute_condition():
  import cv2
  import face_recognition
  #import serial
  import time
  import os
  t=0
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

  # Establish serial communication with Arduino
  #ser = serial.Serial('COM6', 9600)  # Replace 'COM3' with the appropriate serial port

  while t==0:
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
              t=1
              captured_face_encoding = captured_face_encodings[0]
              results = face_recognition.compare_faces(known_faces_encodings, captured_face_encoding)
              if any(results):
                  print("Match found!")
                  cap.release()
                  cv2.destroyAllWindows()
                  new_window = tk.Tk()
                  new_window.geometry("400x400")
                  new_window.configure(background='lavender')
                  def button1():
                    import cv2
                    import os


                    face_cascade = cv2.CascadeClassifier('C:/Users/cubefore/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0/LocalCache/local-packages/Python310/site-packages/cv2/data/haarcascade_frontalface_default.xml')


                    cap = cv2.VideoCapture(0)


                    directory = r'C:\Users\cubefore\OneDrive\Desktop\match testin\known_faces'
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    while True:
    
                        ret, frame = cap.read()

    
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
                        for i, (x, y, w, h) in enumerate(faces):
        
                            face = frame[y:y+h, x:x+w]


                            name = input(f"Enter the roll no of the student {i+1} in the face: ")

        
                            cv2.imwrite(os.path.join(directory, f'{name}.jpg'), face)

        
                            cv2.imshow('face', face)

    
                        cv2.imshow('frame', frame)

    
                        
                        break


                    cap.release()
                    cv2.destroyAllWindows()
                  def button2():
                    import cv2
                    import os


                    face_cascade = cv2.CascadeClassifier('C:/Users/cubefore/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0/LocalCache/local-packages/Python310/site-packages/cv2/data/haarcascade_frontalface_default.xml')


                    cap = cv2.VideoCapture(0)


                    directory = r'C:\Users\cubefore\OneDrive\Desktop\match testin\admin'
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    while True:
    
                        ret, frame = cap.read()

    
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
                        for i, (x, y, w, h) in enumerate(faces):
        
                            face = frame[y:y+h, x:x+w]


                            name = input(f"Enter the name of person {i+1} in the face: ")

        
                            cv2.imwrite(os.path.join(directory, f'{name}.jpg'), face)

        
                            cv2.imshow('face', face)

    
                        cv2.imshow('frame', frame)

    
                        
                        break


                    cap.release()
                    cv2.destroyAllWindows()
                    
                    

                  button1 =tk.Button(new_window, text="add student", width=20, height=2, command=button1, bg="red")
                  button1.pack(pady=10)

                  #button2 =tk.Button(new_window, text="add admin", width=20, height=2, command=button2, bg="red") 
                  #button2.pack(pady=10)

                  #button3 = tk.Button(new_window, text="Button 3", width=10, height=2, bg="red")
                  #button3.pack(pady=10)

                  new_window.mainloop()
                 
              else:
                  print("No match found.")
                 
          else:
              print("No face detected in the captured image.")

      cv2.imshow('Face Detection', frame)

      if cv2.waitKey(1) == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()
    

window = tk.Tk()
window.geometry("400x400")
window.configure(background='lavender')

button1 = tk.Button(window, text="enter attendance", width=20, height=2, command=execute_function, bg="red")
button1.pack(pady=10)

button2 = tk.Button(window, text="add new student", width=20, height=2, command=execute_condition, bg="red")
button2.pack(pady=10)

window.mainloop()
