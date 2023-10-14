import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
path = "pic"
images = []
classNames = []
myList = os.listdir(path)
print(myList) 
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(len(images))
print(classNames)


def Encode(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnow = Encode(images)
print("Ma hoa thanh cong")
print(len(encodeListKnow))

def thamdu(name):
    with open("thamdu.csv", "r+") as f:
        myDatalist = f.readline()
        nameList = []
        for line in myDatalist:
            entry = line.split(",") 
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime("%H:%M:%S:%p")
            f.writelines(f'\n{name},{dtstring}')



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    framS = cv2.resize(frame, (0,0),None,fx = 0.5, fy = 0.5)
    framS = cv2.cvtColor(framS, cv2.COLOR_BGR2RGB)

    facecurFram = face_recognition.face_locations(framS)
    encodecurFram = face_recognition.face_encodings(framS)

    for encodeFace, faceLoc in zip(encodecurFram, facecurFram):
        matches = face_recognition.compare_faces(encodeListKnow, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnow, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
            thamdu(name)
        else:
            name = "Unknown"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 *2, x2*2, y2*2, x1*2
        cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),2)
        cv2.putText(frame, name,(x2,y2), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 2)

    cv2.imshow('Kus face_recognition', frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release() 
cv2.destroyAllWindows() 
