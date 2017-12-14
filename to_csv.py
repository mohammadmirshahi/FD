import dlib,os,cv2,scipy.misc
import numpy as np
num_jitters=input('num_jitters : ')

face_detector = dlib.get_frontal_face_detector()


pose_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

w=open('name.csv','w')
w.write('name\n')
w.close()
w=open('list.csv','w')
for i in range(128):
    w.write(str(i)+',')
i=-2
w.write('name\n')
w.close()
K_name=[]
w_l=open('list.csv','a')
w_n=open('name.csv','a')
a=os.path.join( os.path.dirname(os.getcwd()),'Data')
if not(os.path.exists(a)):
    a=os.path.join( os.getcwd(),'Data')
for r,d,f in os.walk(a):
    i+=1
    print r
    for fi in  f:
        if fi.find('.')!= -1:
            
            img=scipy.misc.imread(r+'/'+fi, mode='RGB')
            K_name.append(r[r.rfind('/')+1:len(r)])
            face_locations = face_detector(img,1)
            raw_landmarks= [pose_predictor(img,rect) for rect in face_locations]

            # How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
            face_encodings=[np.array(face_encoder.compute_face_descriptor(img, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]
            for item in face_encodings[0]:
                w_l.write(str(item)+',')
            w_n.write(r[r.rfind('/')+1:len(r)]+'\n')
            w_l.write(str(i)+'\n')



