import cv2  
  
# np is an alias pointing to numpy library 
import numpy as np 
import math
  
# capture frames from a camera 
cap = cv2.VideoCapture(0) 
scale = 2
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
def angle(pt1,pt2,pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return float((dx1*dx2 + dy1*dy2))/math.sqrt(float((dx1*dx1 + dy1*dy1))*(dx2*dx2 + dy2*dy2) + 1e-10)

# loop runs if capturing has been initialized 
while(1): 
  
    # reads frames from a camera 
    ret, frame = cap.read() 
    frame = cv2.flip(frame, 1)
    # converting BGR to HSV
    blur = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) 
      
    lower = np.array([0,100,100])
    upper = np.array([20,255,255])
    canny = cv2.Canny(frame,80,240,3)
    contours, hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours)):
        #approximate the contour with accuracy proportional to
        #the contour perimeter
        approx = cv2.approxPolyDP(contours[i],cv2.arcLength(contours[i],True)*0.02,True)

        #Skip small or non-convex objects
        if(abs(cv2.contourArea(contours[i]))<100 or not(cv2.isContourConvex(approx))):
            continue

        #triangle
        if(len(approx) == 3):
            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.putText(frame,'TRI',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
        elif(len(approx)>=4 and len(approx)<=6):
            #nb vertices of a polygonal curve
            vtc = len(approx)
            #get cos of all corners
            cos = []
            for j in range(2,vtc+1):
                cos.append(angle(approx[j%vtc],approx[j-2],approx[j-1]))
            #sort ascending cos
            cos.sort()
            #get lowest and highest
            mincos = cos[0]
            maxcos = cos[-1]

            #Use the degrees obtained above and the number of vertices
            #to determine the shape of the contour
            x,y,w,h = cv2.boundingRect(contours[i])
            if(vtc==4):
                cv2.putText(canny,'RECT',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
            elif(vtc==5):
                cv2.putText(canny,'PENTA',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
            elif(vtc==6):
                cv2.putText(canny,'HEXA',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)
            elif(vtc==8):
                cv2.putText(canny,'OCTA',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)

        else:
            #detect and label circle
            area = cv2.contourArea(contours[i])
            x,y,w,h = cv2.boundingRect(contours[i])
            radius = w/2
            if(abs(1 - (float(w)/h))<=2 and abs(1-(area/(math.pi*radius*radius)))<=0.2):
                cv2.putText(canny,'CIRC',(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),2,cv2.LINE_AA)

    #Display the resulting frame
    out.write(frame)
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow("canny", canny)

    # Wait for Esc key to stop 
    k = cv2.waitKey(5) & 0xFF
    if k == 27: 
        break
  
  
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()  
