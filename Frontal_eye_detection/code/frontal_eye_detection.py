#Importing the libraries 
import cv2 as cv
import numpy as np
import os
import pickle
from keras.preprocessing import image 


# Getting the model we created in our jupyter notebook
file = open('/home/ridham/Desktop/model.pkl','rb')
model = pickle.load(file)
file.close()


# Getting he cascade to detect face and eyes
folder=os.path.dirname(__file__)
faces  = cv.CascadeClassifier(os.path.join(folder,'./haarcascade_frontalface_default.xml'))
eyes = cv.CascadeClassifier(os.path.join(folder,'./haarcascade_eye.xml'))

# Using videoCapture to capture my face through facecam
vid = cv.VideoCapture(0)
counter = 0
count =10
timer = 0
attempts = 3
counter_eyes = 0
count_eyes =10
timer_eyes = 0
attempts = 3



#Looping all the things
while True:
    status,img = vid.read()
    gray_face = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face = faces.detectMultiScale(gray_face,1.9,5)
    if len(face) > 0:
        for (x,y,w,h) in face:


            # Adding rectangle to the detected face
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            eyes_gray = gray_face[y:y+h,x:x+w]
            eyes_img = img[y:y+h,x:x+w]
            eye = eyes.detectMultiScale(eyes_gray,1.8,5)
            if len(eye) > 1:
                for (ex,ey,ew,eh) in eye:  
                    img1 = eyes_gray[ey:ey+eh,ex:ex+ew]
                    img1= image.img_to_array(img)
                    img1 = np.array([img])
                    img1.resize((1,45,45,1))
                    
                    
                    
                    
                    # When the eyes are detected they are checked by the model if they are in the frontal orientation 
                    if model.predict(img1)[0][0] > 0.3:

                        # Adding rectangle to the detected eye
                        cv.rectangle(eyes_img,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                        cv.imshow('IMG',img)
                    

                    counter_eyes = 0
                    count_eyes = 10
            else:

                # To display when the eyes are not detected
                cv.putText(img,f'look at the screen',(313,255),cv.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),2,cv.LINE_AA)
                counter_eyes += 1
                if counter > 500:
                    cv.putText(img,f'look at the screen or else you will be disqualified {count_eyes} sec',(60,420),cv.FONT_HERSHEY_TRIPLEX, .5,(0,0,255),2,cv.LINE_AA)
                timer_eyes +=1
                if timer_eyes == 50:
                    timer_eyes = 0
                    count-=1
                if count_eyes == 0:
                    break
                
                if count_eyes == 5 and timer_eyes == 50:
                    attempts -=1
                if attempts == 0:
                    break

        counter = 0
        count = 10            
    else:


        # To display when the face is not detected
        cv.rectangle(img,(350,290),(470,310),(0,0,255),-1)
        cv.putText(img,f'Not detected',(363,305),cv.FONT_HERSHEY_SIMPLEX, .5,(255,255,255),2,cv.LINE_AA)
        counter +=1
        if counter > 500:
            cv.rectangle(img,(55,400),(600,440),(0,0,255),1)
            cv.putText(img,f'You will be disqualified if you dont return to exam in {count} sec',(60,420),cv.FONT_HERSHEY_TRIPLEX, .5,(0,0,255),2,cv.LINE_AA)
            timer +=1
            if timer == 50:
                timer = 0
                count-=1
            if count == 0:
                break
            
            if count == 5 and timer == 50:
                attempts -=1
            if attempts == 0:
                break



    # Displaying the final image
    cv.imshow('IMG',img)
    if cv.waitKey(1) == ord('q'):
        break
vid.release()
cv.destroyAllWindows()
