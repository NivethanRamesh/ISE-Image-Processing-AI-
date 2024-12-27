## Lecture 2(Kernel and convolutional)
import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lena.png")
gs = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
k = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=np.uint8)
res = cv2.filter2D(gs,-1,k)  # Convolution
cv2.imshow("Lena",gs)
cv2.imshow("Filtered",res)
cv2.waitKey(0)


mu = 5
sig = 0.6
x= np.linspace(0,10,100)
y=np.exp(-(x-mu)**2/2/sig**2)
plt.figure(0)
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(x,y,label="Gaussian")
plt.legend()
plt.show()
