from glob import glob
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import threading
import numpy as np
from tensorflow import keras



model = keras.models.load_model('model_optimal.h5')

emotion_dict = {
    0 : 'angry',
    1 : 'digust',
    2 : 'fear',
    3 : 'happy',
    4 : 'neutral',
    5 : 'sad' ,
    6 : 'surprised',
    7 : ''
}


cur_path = os.path.dirname(os.path.abspath(__file__))
cur_path = cur_path[:-4]
emoji_dict ={
    0 : cur_path + "/Data/Emojies/angry.png",
    1 : cur_path + "/Data/Emojies/disgust.png",
    2 : cur_path + "/Data/Emojies/fear.png",
    3 : cur_path + "/Data/Emojies/happy.png",
    4 : cur_path + "/Data/Emojies/neutral.png",
    5 : cur_path + "/Data/Emojies/sad.png",
    6 : cur_path + "/Data/Emojies/surprised.png",
    7 : cur_path + "/Data/Emojies/default.png"
}




prediction = 7


width, height = 500, 500
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = tk.Tk()
root.geometry("800x500")
root.bind('<Escape>', lambda e: root.quit())
cam = tk.Label(root,bd=4)
emoji  = tk.Label(root,bd=4)
text =tk.Label(root,bd=10,fg="#CDCDCD",bg='black')
cam.pack(side =LEFT)
text.pack()
emoji.pack(side =RIGHT)


event  = threading.Event()


def on_closing():
    event.set()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_closing)

def show_frame():
    while True:
        _, frame = cap.read()
        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            cropped_img = gray_frame[y:y + h, x:x + w]
            cropped_img = cv2.resize(cropped_img, (48, 48))
            cropped_img = np.array(cropped_img)
            cropped_img = cropped_img.reshape(1,48,48,1)
            cropped_img = cropped_img/255.0 
            global prediction
            prediction = np.argmax(model.predict(cropped_img))

        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        cam.imgtk = imgtk
        cam.configure(image=imgtk)
        if event.is_set() :
            break
            #cam.after(10, show_frame)

def show_avatar():
    frame2=cv2.imread(emoji_dict[prediction])
    img2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(img2)
    imgtk=ImageTk.PhotoImage(image=img2)
    emoji.imgtk=imgtk
    text.configure(text=emotion_dict[prediction],font=('arial',45,'bold'))
    emoji.configure(image=imgtk)
    if not event.is_set() :
        emoji.after(10, show_avatar)

threading.Thread(target=show_frame).start()
threading.Thread(target = show_avatar).start()


root.mainloop()