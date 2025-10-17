import cv2
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
import os

def capture_photo():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Error", "Failed to capture photo.")
        return

    
    cv2.imshow('Captured Photo', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    save_path = r"C:\Users\cubefore\OneDrive\Desktop\match testin\ac\captured_photo.png"
    cv2.imwrite(save_path, frame)
    messagebox.showinfo("Success", f"Photo saved to {save_path}")

def launch_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam.")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        messagebox.showerror("Error", "Failed to capture photo.")
        return

    
    cv2.imshow('Captured Photo', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    save_path = r"C:\Users\cubefore\OneDrive\Desktop\match testin\ac\captured_photo.png"
    cv2.imwrite(save_path, frame)
    messagebox.showinfo("Success", f"Photo saved to {save_path}")
    
    root = tk.Tk()
    root.title("Photo Capture")

    capture_button = tk.Button(root, text="Capture Photo", command=capture_photo)
    capture_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    launch_camera()
