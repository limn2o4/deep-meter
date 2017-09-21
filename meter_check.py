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
x = cv2.Sobel(binary,cv2.CV_16S,1,0)
y = cv2.Sobel(binary,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
edges = cv2.addWeighted(absX,0.5,absY,0.5,0)
#use soble not canny
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,300,10)
circles = cv2.HoughCircles(gary,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=50)
radius = 0
maxx,maxy = 0,0
for i in circles[0,:]:
    # draw the outer circle
    #cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    if i[2] > radius :
        radius = i[2]
        maxx = i[0]
        maxy = i[1]
    # draw the center of the circle
    #cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
cv2.circle(img,(maxx,maxy),radius,(0,255,0),2)
cv2.imshow("333",edges)
for x1,y1,x2,y2 in lines[0]:
    #print(line[0][1],line[0][0])
    #x1,y1,x2,y2 = lis[0][0],lis[0][1],lis[0][2],lis[0][3]
   # print(x1,y1,x2,y2)
    print(np.arctan((x1 - x2)/(y1 - y2)))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow("123",img)
cv2.waitKey(0)