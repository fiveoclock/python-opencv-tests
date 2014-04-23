import numpy as np
import cv2
from matplotlib import pyplot as plt


cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # global equalize Histogram
    equ = cv2.equalizeHist(gray)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = clahe.apply(gray)

    gray_orig_equ = np.hstack((gray,clahe,equ)) #stacking images side-by-side


    # calculate histogram
    h = np.zeros((300,256,3))

    bins = np.arange(256).reshape(256,1)
    color = [ (255,0,0),(0,255,0),(0,0,255) ]

    for ch, col in enumerate(color):
        hist_item = cv2.calcHist([frame],[ch],None,[256],[0,255])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.column_stack((bins,hist))
        cv2.polylines(h,[pts],False,col)

    h=np.flipud(h)
    cv2.imshow('colorhist',h)

    # Display the resulting frame
    cv2.imshow('color',frame)
    cv2.imshow('frame',gray_orig_equ)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

