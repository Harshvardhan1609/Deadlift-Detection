import tkinter as tk
import customtkinter as ck

import pandas as pd
import numpy as np
import pickle

import mediapipe as mp
import cv2
from PIL import Image, ImageTk
from landmarks import landmarks

# From here we will do the work of creating an gui for our deadlift detection app

window = tk.Tk()
window.geometry("480x700")
window.title("Deadlift Detection")
ck.set_appearance_mode("dark")

# setting up the class label and then placing the text and configuring the text.

classLabel = ck.CTkLabel(window, height=40, width=120, font=(
    "Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='STAGE')
counterLabel = ck.CTkLabel(window, height=40, width=120, font=(
    "Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS')
probLabel = ck.CTkLabel(window, height=40, width=120, font=(
    "Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB')

# Creating the box with the same element but the only difference is here we will now add the backgorund color as black and also we will create our text white in color.

classBox = ck.CTkLabel(window, height=40, width=120, font=(
    "Arial", 20), text_color="white", fg_color="black")
classBox.place(x=10, y=41)
classBox.configure(text='0')
counterBox = ck.CTkLabel(window, height=40, width=120, font=(
    "Arial", 20), text_color="white", fg_color="black")
counterBox.place(x=160, y=41)
counterBox.configure(text='0')
probBox = ck.CTkLabel(window, height=40, width=120, font=(
    "Arial", 20), text_color="white", fg_color="black")
probBox.place(x=300, y=41)
probBox.configure(text='0')

# Now we will reset the counter and inside this we will create counter as  global and initialize the counter with the value of 0 and this we will trigger with the button which we will create in our next step.


def reset_counter():
    global counter
    counter = 0


# Now it is the time for placing the button and this we can do easily by using the Custom tkinter library and we will set it at the desirable coordinates

button = ck.CTkButton(window, text="RESET", height=40, width=120, font=(
    "Arial", 20), text_color="white", fg_color="black", command=reset_counter )
button.place(x=10, y=600)

# Now we will create frame and we will mention the height and width and then we will place it at the desirable coordinates and then we will add the label to it and then we will place it

frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90)
lmain = tk.Label(frame)
lmain.place(x=0, y=0)


# Now from here we will use the mediapipe by google to draw_utils on our pose and then we will give the min_tracking_confidence and min_detection_confidence values as 0.5 in out test phase 1

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

# Now we will open the Pickle file and load it in model and this we will do by the following code :

with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

# Now it is the time for cv2 to capture the video and then we will use the videoCapture(web_cam_no) to start capturing the video then we will make the current_stage , bodylang_class as empty string counter as 0 and in the bodylang_prob we will add the np.array([0,0])

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0
bodylang_prob = np.array([0, 0])
bodylang_class = ''

# PHASE 1


def detect():
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob

    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
    mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

    # Phase 2

    try:
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns = landmarks) 
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0] 

        if bodylang_class =="down" and bodylang_prob[bodylang_prob.argmax()] > 0.7: 
            current_stage = "down" 
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up" 
            counter += 1 


    except Exception as e:
        print(e)

    # Phase 3

    img = image[:, :460, :]
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, detect)

    # Phase 4

    counterBox.configure(text=counter)
    probBox.configure(text=bodylang_prob[bodylang_prob.argmax()])
    classBox.configure(text=current_stage)


detect()
window.mainloop()
