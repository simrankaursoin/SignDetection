import numpy as np
import cv2
import psutil

cap = cv2.VideoCapture(0)

## some videowriter props
sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = 20
#fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
#fourcc = cv2.VideoWriter_fourcc('m', 'p', 'e', 'g') 
fourcc = cv2.VideoWriter_fourcc(*'mpeg') 

## open and set props
vout = cv2.VideoWriter()
vout.open('output.mp4',fourcc,fps,sz,True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    vout.write(frame) 

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower = np.array([0,100,100])
    upper = np.array([20,255,255])

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    print(psutil.virtual_memory())
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break


cap.release()
vout.release()
cv2.destroyAllWindows()
