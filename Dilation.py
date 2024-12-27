import cv2
import numpy as np






## PART 1

def f(x):
    print(x)
    return(x)

a = cv2.imread("blob-1.jpg")
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

cv2.namedWindow("Control")
cv2.createTrackbar("Iter","Control",1,20,f)

while(True):
    iter = cv2.getTrackbarPos("Iter","Control")
    res = cv2.morphologyEx(a,cv2.MORPH_DILATE,ker,iterations=iter)
    cv2.imshow("Dilate",res)

    iter = cv2.getTrackbarPos("Iter", "Control")
    res = cv2.morphologyEx(a, cv2.MORPH_CLOSE, ker, iterations=iter)
    cv2.imshow("Close", res)

    iter = cv2.getTrackbarPos("Iter", "Control")
    res = cv2.morphologyEx(a, cv2.MORPH_ERODE, ker, iterations=iter)
    cv2.imshow("Erode", res)

    iter = cv2.getTrackbarPos("Iter", "Control")
    res = cv2.morphologyEx(a, cv2.MORPH_OPEN, ker, iterations=iter)
    cv2.imshow("Open", res)

    key=cv2.waitKey(5)
    if(key==32 or iter==-1):
        break


## PART 2

import cv2
import numpy as np


a = cv2.imread("shapes1.png")
b=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
r,c = cv2.threshold(b,100,255,cv2.THRESH_BINARY_INV)
print(r)
cv2.imshow("Orig",a)
cv2.imshow("Grey",b)
cv2.imshow("B&W",c)
cv2.waitKey(0)

## PART 3

a=cv2.imread("coin.jpg")
b=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
# b=cv2.GaussianBlur(b,(5,5),1)
r,c = cv2.threshold(b,80,255,cv2.THRESH_BINARY)

c = cv2.morphologyEx(c,cv2.MORPH_DILATE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=1)
cv2.imshow("Orig",a)
cv2.imshow("Window",c)
cv2.waitKey(0)