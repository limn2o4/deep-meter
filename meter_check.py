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


img = cv2.imread("D:/project/deep-meter/meter0.jpg")
gary = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#binary = cv2.adaptiveThreshold(gary,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,10)
ret,binary = cv2.threshold(gary,100,255,cv2.THRESH_BINARY)
#mn = minfilter(binary,3)

binary = cv2.blur(binary,(3,3))
cv2.imshow("out",binary)
cv2.waitKey(0)
x = cv2.Sobel(binary,cv2.CV_16S,1,0)
y = cv2.Sobel(binary,cv2.CV_16S,0,1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
edges = cv2.addWeighted(absX,0.5,absY,0.5,0)
#use soble not canny
#edges = cv2.Canny(binary,100,300)

circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,10,200,param1=50,param2=50)
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
roi = edges[int(maxx - radius):int(maxx + radius),int(maxy-radius):int(maxy+radius)]
cv2.imshow("1234",roi)
cv2.waitKey(0)
lines = cv2.HoughLinesP(roi,1,np.pi/180,100,300,20)
for x1,y1,x2,y2 in lines[0]:
    #print(line[0][1],line[0][0])
    #x1,y1,x2,y2 = lis[0][0],lis[0][1],lis[0][2],lis[0][3]
   # print(x1,y1,x2,y2)
    print(180/np.pi*np.arctan((x1 - x2)/(y1 - y2)))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imshow("123",img)
cv2.waitKey(0)