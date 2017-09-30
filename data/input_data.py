import cv2
import numpy as np
import os
import sys
data_num = {}
data_img = {}
data_size = 0
def get_number():
    path = os.getcwd()
    file_path = os.path.abspath(os.path.join(path,"data/","numbers.txt"))
    #print("file in:"+file_path)
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            idx,num = line.split(':')
            data_num.update({int(idx):int(num)})

def get_image():
    pwd  = os.getcwd()
    path = os.path.join(pwd,"data");
    #print(path)
    images = os.listdir(path)
    #print(images)
    for i,name in enumerate(images):
        image = cv2.imread(os.path.join(path,name),0)

        if image is None:
            continue

        #print(i,name,image)
        image = cv2.resize(image,(160,60),interpolation=cv2.INTER_LINEAR)
        data_img.update({int(i):image})
        #print(image)
    #print(path)
def get_size():
    return len(data_img)
def get_data(idx = 1):

    return data_num[idx],data_img[idx];

if __name__ == "data.input_data":
    print("load data......")
    #print(sys.path[0])
    get_number()
    #print(data_num)
    get_image()
    #print(data_img)
    print("complete!")
