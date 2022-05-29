import cv2
import numpy as np 
import face_recognition
import os
from datetime import datetime

path='images'
images=[]
classNames=[]
mylist=os.listdir(path)
for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def FindEncodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markAttendance(name):
    with open('attendance1.csv','r+') as f:
        myDataList=f.readlines()
        namelist=[]
        for line in myDataList:
            entry=line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtstring=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')




encodeListknown=FindEncodings(images)
print('Encoding complete')

cap=cv2.VideoCapture(0)

while True:
    success,img= cap.read()
    imgS=cv2.resize(img,None,0.25,0.25)
    img=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    faceCurFrames=face_recognition.face_locations(imgS)
    encodeCurFrames=face_recognition.face_encodings(imgS,faceCurFrames)

    for encodeFace,faceloc in zip(encodeCurFrames,faceCurFrames):
         matches=face_recognition.compare_faces(encodeListknown,encodeFace)
         facedis=face_recognition.face_distance(encodeListknown,encodeFace)
         matchIndex=np.argmin(facedis)

         if matches[matchIndex]:
             name=classNames[matchIndex].upper()
             y1,x2,y2,x1=faceloc
             y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
             cv2.rectangle(img(x1,y1),(x2,y2),(90,255,0),2)
             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
             markAttendance(name)
    cv2.imshow('webcam',img)
    cv2.waitKey(0)
    
















































































































































































































































































































   

imgElon=face_recognition.load_image_file("Elon musk.jpg")
imgElon=cv2.cvtColor(imgElon.cv2.COLOR_BGR2RGB)
imgTest=face_recognition.load_image_file("Elon test.jpg")
imgTest=cv2.cvtColor(imgElon.cv2.COLOR_BGR2RGB)