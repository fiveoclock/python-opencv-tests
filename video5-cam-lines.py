import numpy as np
import cv2
import cv2.cv as cv

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE - equalize Histogram
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = clahe.apply(gray)
    edges = cv2.Canny(clahe,50,150,apertureSize = 3)

    """
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    #if (lines is not None) and (len(lines) != 0):
    if (lines is None) or (len(lines) == 0):
        pass
    else:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 2)
    """

    # line detect
    minLineLength = 600
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

    if (lines is None) or (len(lines) == 0):
        pass
    else:
        for x1,y1,x2,y2 in lines[0]:
            cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)

    detect = np.hstack((edges,clahe)) #stacking images side-by-side



    # circle detect
    img = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,80,
                            param1=220,param2=30,minRadius=0,maxRadius=90)

    if (circles is None) or (len(circles) == 0):
        pass
    else:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)



    # Display the resulting frame
    cv2.imshow('frame', detect)
    cv2.imshow('color',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

