import cv2
import numpy as np
import os
c='z-haarcascade_frontalface_default.xml'
#c='z-haarcascade_frontalface_alt_tree.xml'
#c='z-haarcascade_frontalface_alt.xml'

path=os.getcwd()
face=cv2.CascadeClassifier(c)
path=os.path.dirname(path)
c=cv2.VideoCapture(0)
if not(os.path.exists(path+'/Data/f')):
    os.mkdir(path+'/Data/f')
i=1
while True:
    ret,f=c.read()
    gr=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gr,1.3,5)
    for (x,y,w,h) in faces:
        i+=1
        fa=f[y:y+h,x:x+w];
        cv2.imwrite(path+'/Data/f/'+str(i)+'.jpg',fa)
        cv2.rectangle(f,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('f',f)

    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
    elif k==ord('q'):
        cv2.destroyAllWindows()

c.release()
cv2.destroyAllWindows()
