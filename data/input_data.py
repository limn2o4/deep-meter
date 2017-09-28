import cv2
import numpy as np
import os
data_num = {}
data_img = {}
data_size = 0
def get_number():
    with open("numbers.txt") as file:
        lines = file.readlines()
        for line in lines:
            idx,num = line.split(':')
            num = int(num)
            data_num.update({int(idx):num})

def get_image():
    path  = os.getcwd()
    images = os.listdir(path)
    for i,image_path in enumerate(images):
        image = cv2.imread(image_path,0)
        data_img.update({int(i):image})

    #print(path)
def get_size():
    return len(data_img)
def get_data(idx = 1):

    return data_num[idx],data_img[idx];

if __name__ == "input_data.py":
    print("load data......")
    get_number()
    #print(data_num)
    get_image()
    #print(data_img)
    print("complete!")