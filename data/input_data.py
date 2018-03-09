import cv2
import numpy as np
import random
import os
import csv
org_data = {}
org_num = {}
label = {}
data_size = 0
def load_data(file_path):
    #print("file in:"+file_path)
    with open(file_path,newline='') as file:
        reader = csv.reader(file)
        #data_size = len(lines)
        for i,row in enumerate(reader):
            idx = int(row[0])
            num = int(row[1])
            if num >= 10:
                continue
            org_data.setdefault(int(num),[]).append(int(idx))
            label[idx] = num % 10
    print([len(size) for size in org_data.values()])

def next_batch(batch_size = 100):
    batch_x = np.zeros([batch_size,1024])
    batch_y = np.zeros([batch_size,10])
    for i in range(0, batch_size):
        idx = random.randint(0,24000)
        if idx not in label:
            i -= 1
            continue
        #print(idx)
        img = cv2.imread("./data/image/" + str(idx) + ".jpg", 0)
        batch_x[i:] = img.flatten() / 255
        batch_y[i][label[idx]] = 1

    return [batch_x,batch_y]

def next_test_batch(batch_size = 100):
    batch_x = np.zeros([batch_size,1024])
    batch_y = np.zeros([batch_size,10])
    cnt  = 0
    # for num in range(0,10):
    #     #print(len(org_data[num]))
    #     for times in range(batch_size//10):
    #         idx = org_data[num][random.randint(0, len(org_data[num])-1)]
    #         #print(idx)
    #         img = cv2.imread("./data/image/" + str(idx) + ".jpg", 0)
    #         batch_x[cnt:] = img.flatten() / 255
    #         batch_y[cnt][num] = 1
    #         cnt += 1
    #print(batch_y)
    for i in range(0,batch_size):
        idx = random.randint(24000,30000+1)
        #print(idx)
        if idx not in label:
            i -= 1
            continue
        img = cv2.imread("./data/image/"+str(idx)+'.jpg',0)
        batch_x[i:] = img.flatten()/255
        batch_y[i][label[idx]] = 1

    return [batch_x,batch_y]

if __name__ == "__main__":
    print('------loading------\n')
    load_data("label.csv")
    next_batch(100)