import cv2
import face_recognition
import os
import numpy as np
from tkinter import *
from tkinter import messagebox, simpledialog
from datetime import datetime

# ===================== GLOBAL =====================
path = 'ImagesAttendance'
if not os.path.exists(path):
    os.makedirs(path)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def loadKnownFaces():
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    encodeListKnown = findEncodings(images)
    return encodeListKnown, classNames

def markLogin(name):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    with open('log.csv', 'a') as f:
        f.write(f'{name},{dtString}\n')

# ===================== LOGIN FUNCTION =====================
def faceLogin():
    try:
        encodeListKnown, classNames = loadKnownFaces()
    except:
        messagebox.showerror("Error", "Gagal memuat wajah. Register dulu!")
        return

    cap = cv2.VideoCapture(0)
    success = False

    while True:
        ret, frame = cap.read()
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex]
                markLogin(name)
                messagebox.showinfo("Login Berhasil", f"Selamat datang, {name}!")
                success = True
                break

        if success:
            break

        cv2.imshow('Login Wajah - Tekan q untuk batal', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== REGISTER FUNCTION =====================
def registerFace():
    name = simpledialog.askstring("Nama", "Masukkan nama lengkap:")
    if not name:
        return

    cap = cv2.VideoCapture(0)
    count = 1

    while True:
        ret, img = cap.read()
        cv2.imshow("Ambil Wajah - Tekan 's' untuk simpan, 'q' untuk keluar", img)

        key = cv2.waitKey(1)
        if key == ord('s'):
            filename = f"{path}/{name}.jpg"
            cv2.imwrite(filename, img)
            messagebox.showinfo("Sukses", f"Wajah {name} berhasil direkam!")
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ===================== GUI =====================
def logout():
    root.destroy()

root = Tk()
root.title("Face Recognition Login")
root.geometry("400x300")

Label(root, text="Face Recognition App", font=("Helvetica", 16, "bold")).pack(pady=20)

Button(root, text="Login", width=20, height=2, command=faceLogin).pack(pady=10)
Button(root, text="Register", width=20, height=2, command=registerFace).pack(pady=10)
Button(root, text="Keluar", width=20, height=2, command=logout).pack(pady=10)

root.mainloop()
