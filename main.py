import cv2
import numpy as np

from source.using_harrcascad import harrCascade


face_harr= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


while True:

    ret,img = cap.read()
    
    blurred_img = harrCascade.bluring(img)
    
    # gray_frame = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # frames_faces = face_harr.detectMultiScale(gray_frame,1.2,4)
    
    
    # for (x,y,w,h) in frames_faces:
        
    #     x,y,w,h=x,y,w-40,h-40
    #     kernel= np.ones((h,w),np.float32)/(h*w)
    #     img[y:y+h,x:x+w] = cv2.filter2D(img[y:y+h,x:x+w],-1,kernel)
        
       
        
        
        
        
    
    
    
    # print(bounded_pixels)
    cv2.imshow("img",blurred_img)
    
    
    k=cv2.waitKey(20) & 0xFF
    
    if k == ord("c"):
        break
    
        
cap.release()
cv2.destroyAllWindows()    

