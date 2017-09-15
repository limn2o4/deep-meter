import cv2
import numpy as np
import argparse 

#parser = argparse.ArgumentParser()

#check if the light in roi is on
def drawHist(hist):
    minV,maxV,minL,maxL = cv2.minMaxLoc(hist)
    histImg = np.zeros([256,256,3],np.uint8)
    phi = int(0.9*256)

    for h in range(256):
        intens = int(hist[h]*phi/maxV)
        cv2.line(histImg,(h,256),(h,256 - intens),[255,255,255])
    return histImg


img = cv2.imread("D:\project\deep-meter\led2.jpg")
r,g,b = cv2.split(img)
h,w,c = np.shape(img)
_,b_th = cv2.threshold(b,200,255,cv2.THRESH_BINARY)
_,g_th = cv2.threshold(g,200,255,cv2.THRESH_BINARY)
_,r_th = cv2.threshold(r,200,255,cv2.THRESH_BINARY)
res = g_th - r_th;

cv2.imshow("g",res)
cv2.imshow("123",)
cv2.waitKey(0)
