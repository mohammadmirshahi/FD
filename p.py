import  cv2,time
import numpy as np
import pandas as pd
import dlib
#from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv('list.csv')


data=data[['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19',
        '20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40',
        '41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61',
        '62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82',
        '83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103',
        '104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121',
        '122','123','124','125','126','127']]



name=pd.read_csv('name.csv')


face_detector = dlib.get_frontal_face_detector()


predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

config=open('config.txt','r').read().split('\n')


c=cv2.VideoCapture(int(config[0]))
num_jitters=int(config[1])

import socket
UDP_IP = config[2]
UDP_PORT = 5005
limit=float(config[3])
sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) 
plis_w=open('plist.lst','a')
p_list=[]
while True:
    if len(p_list)>100:
            plis_w.writelines(p_list[0]);
            p_list.remove(p_list[0])
    ret,f=c.read()
    gr=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    rects = face_detector(gr, 0)
    for rect in rects:

        shape= predictor(f,rect)

        #num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
        face_encodings=np.array(face_encoder.compute_face_descriptor(f, shape, num_jitters)) 
        a=np.linalg.norm(data - face_encodings, axis=1)

        mn=min(a)
        fi=''
        if mn<=limit:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fi=name['name'][np.where(a==mn)[0][0]]
            cv2.putText(f,fi,(dlib.rectangle.left(rect),dlib.rectangle.top(rect)),font,0.5,(0,255,255),1)
            sock.sendto(fi, (UDP_IP, UDP_PORT))
            sock.sendto(fi, (UDP_IP, UDP_PORT+1))
            if p_list.count(fi)<1:
                p_list.append(fi)

    cv2.imshow('f',f)
    f=None
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break
    elif k==ord('q'):
        cv2.destroyAllWindows()

for persion in p_list:
    plis_w.writelines(persion)
plis_w.writelines('\n'+str(time.localtime()))
plis_w.close()
c.release()
cv2.destroyAllWindows()
