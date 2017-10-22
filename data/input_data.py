import cv2
import numpy as np
import random
import os
org_data = {}
data_img = {}
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
            org_data.update({int(idx):int(num)})
    return data_size

def next_batch(batch_size = 128):
    batch_x = np.zeros([batch_size,1024])
    batch_y = np.zeros([batch_size,10])
    for i in range(batch_size):
        idx = random.randint(1, data_size - 1)

        if org_data[idx] != 10:
            #print(idx,org_data[idx])
            img = cv2.imread("./data/image/" + str(idx) + ".jpg", 0)
            batch_x[i:] = img.flatten()/255
            batch_y[i][org_data[idx]] = 1
    return [batch_x,batch_y]


if __name__ == "data.input_data":
    print("......load data......")
    data_size = load_data()
    print("complete! size = {}".format(data_size))
    #print(sys.path[0])
