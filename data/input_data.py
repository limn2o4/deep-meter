import cv2
import numpy as np
import random
import os
org_data = {}
org_num = {}
data_size = 0
def load_data():
    path = os.getcwd()
    file_path = os.path.abspath(os.path.join(path,"data/","numbers.txt"))
    #print("file in:"+file_path)
    with open(file_path) as file:
        lines = file.readlines()
        data_size = len(lines)
        for i,line in enumerate(lines):
            idx,num = line.split(':')
            org_data.setdefault(int(num),[]).append(int(idx))
    print([len(size) for size in org_data.values()])
    return data_size

def next_batch(batch_size = 100):
    batch_x = np.zeros([batch_size,1024])
    batch_y = np.zeros([batch_size,10])
    cnt  = 0
    for num in range(0,10):
        #print(len(org_data[num]))
        for times in range(batch_size//10):
            idx = org_data[num][random.randint(0, len(org_data[num])-1)]
            #print(idx)
            img = cv2.imread("./data/image/" + str(idx) + ".jpg", 0)
            batch_x[cnt:] = img.flatten() / 255
            batch_y[cnt][num] = 1
            cnt += 1
    #print(batch_y)
    return [batch_x,batch_y]
def next_batch_by_num(batch_size = 128,num = 5):
    batch_x = np.zeros([batch_size,1024])
    batch_y = np.zeros([batch_size,10])
    for i in range(batch_size):
        #idx = org_data[num][random.randint(0,len(org_data[num])-1)]
        idx = org_data[num][i]
        #print(idx)
        img = cv2.imread("./data/image/"+str(idx)+".jpg",0)
        batch_x[i:] = img.flatten() / 255
        batch_y[i][num] = 1
    #print(batch_x)
    return [batch_x,batch_y]

