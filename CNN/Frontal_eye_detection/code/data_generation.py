# For creation of data set we will be using this code





#Import these libraries
import cv2 as cv
import numpy as np
import os


#Import the cacades
folder=os.path.dirname(__file__)
faces  = cv.CascadeClassifier(os.path.join(folder,'./haarcascade_frontalface_default.xml'))
eyes = cv.CascadeClassifier(os.path.join(folder,'./haarcascade_eye.xml'))
vid = cv.VideoCapture(0)
counter = 0
images = 'eyes_train'

path = 'data'
images = ['eyes_test','eyes_train']
for image in images:
    destination = path +  '/' + image + '/' + 'front' 
    if  os.path.isdir(destination) == False:
        os.makedirs(destination)



while True:
    status,img = vid.read()
    gray_face = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face = faces.detectMultiScale(gray_face,1.7,5)
    if len(face) > 0:
        for (x,y,w,h) in face:
            cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
            eyes_gray = gray_face[y:y+h,x:x+w]
            eyes_img = img[y:y+h,x:x+w]
            eye = eyes.detectMultiScale(eyes_gray,1.8,5)
            
            if len(eye) > 1:
                 for (ex,ey,ew,eh) in eye:
                    
                    if counter <=1000:
                        cv.imwrite(f'data/{images}/front/Front{counter}.jpg',eyes_gray[ey:ey+eh,ex:ex+ew])
                        cv.rectangle(eyes_img,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
                        counter +=1
                    else:
                        images = 'eyes_test'
                        counter = 0
                        

    cv.imshow('IMG',img)


    #READ IT VERY IMPORTANT!!!!!!
    # the time you take to creat the number of images may vary so keep checking the folder were the images are been stored and press "q" key to terminate the code
    if cv.waitKey(30) == ord('q') & 0xff:
        break
vid.release()
cv.destroyAllWindows()
