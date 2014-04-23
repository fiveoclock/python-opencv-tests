import cv2
import numpy as py

img = cv2.imread('road1.jpg')


px = img[100,100]
print px

# accessing only blue pixel
blue = img[100,100,0]
print blue

cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()

