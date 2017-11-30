import cv2
import numpy as np
import random
import os
import csv
org_data = {}
org_num = {}
data_size = 0
def load_data():
    path = os.getcwd()
    file_path = os.path.abspath(os.path.join(path,'data',"lable.csv"))
    #print("file in:"+file_path)
    with open(file_path,newline='') as file:
        reader = csv.reader(file)
        #data_size = len(lines)
        for i,row in enumerate(reader):
            idx = row[0]
            num = row[1]
            org_data.setdefault(int(num),[]).append(int(idx))
            data_size = i
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
def next_batch_by_num(batch_size,num):
    batch_x = np.zeros([batch_size,1024])
    batch_y = np.zeros([batch_size,10])
    for i in range(batch_size):
        idx = org_data[num][random.randint(0,len(org_data[num])-1)]
        #idx = org_data[num][i]
        #print(idx)
        img = cv2.imread("./data/image/"+str(idx)+".jpg",0)
        batch_x[i:] = img.flatten() / 255
        batch_y[i][num] = 1
    #print(batch_x)
    return [batch_x,batch_y]

if __name__ == "__main__":
    print('------loading------\n')
    load_data()