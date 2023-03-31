import cv2
import numpy as np


class harrCascade:
    
    face_harr = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    def __init__(self,frame):
        self.frame = frame
        
        
    def bluring(self):
        
        grey_img = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)
        
        faces = self.face_harr.detectMultiScale(grey_img,1.1,4)
        
        for (x,y,w,h) in faces:
            kernel =  np.ones((h,w),np.float32)/(h*w)
            
            self.frame[y:y+h,x:x+w] = cv2.filter2D(self.frame[y:y+h,x:x+w],-1,kernel)
            
            return self.frame
            
        
            
        
        