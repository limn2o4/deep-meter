import cv2
import numpy as np
def minfilter(image,ksize = 3):
    dst = np.zeros(image.shape,np.int8)
    h,w = image.shape
    for i in range(1,h-1):
        for j in range(1,w -1):
            minx = 0
            for i in range(i-1,i+1):
                for m in range(j-1,j+1):
                    if image[i,j] > minx:
                        minx = image[i,j]
            dst[i,j] = minx
    return dst


img = cv2.imread(r"D:\project\deep-meter\meter1.jpg")
gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
binary = cv2.adaptiveThreshold(gary,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,10)
mn = minfilter(binary,3)
#cv2.imshow("out",mn)
binary = cv2.blur(binary,(3,3))
binary = cv2.
edges = cv2.Canny(binary,100,200)
#use soble not canny
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,200,10)
cv2.imshow("333",edges)
for lis in lines:
    #print(line[0][1],line[0][0])
    x1,y1,x2,y2 = lis[0][0],lis[0][1],lis[0][2],lis[0][3]
   # print(x1,y1,x2,y2)
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow("123",img)
cv2.waitKey(0)