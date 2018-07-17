import tensorflow as tf
from keras.models import Model
from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Reshape,Input,Activation,Lambda
from keras.optimizers import SGD,Adagrad,Adadelta
from keras.regularizers import l2
import numpy as np
import argparse
import data.input_data as data
import random
import time
import cv2
#version 2 leNet-5
def build_network():

    X = Input(shape=(28,28,1))

    conv1 = Conv2D(filters = 6,kernel_size = (5,5),padding = 'valid',activation = 'tanh')(X)
    pool1 = MaxPool2D(kernel_size = (2, 2))(conv1)
    conv2 = Conv2D(filters = 16,kernel_size = (10, 10),padding = 'valid',acitvation = 'tanh')(pool1)
    pool2 = MaxPool2D(Kernel_size = (2,2))(conv2)
    flat = Flatten()(pool2)
    fc1 = Dense(120,activation = 'tanh')(flat)
    fc2 = Dense(48,activation='tanh')(fc1)
    output = Dense(10,activation='softmax')(fc2)
    model = Model(inputs = x,outputs = output)
    return model


def train_network():
    # TODO:need load data
    sgd_optmizer = SGD(lr=0.05,momentum=0.9,decay=1e-6,nesterov=True)
    model = build_network()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=sgd_optmizer,
        metrics=['accuracy']
    )


def rec_number():
    #TODO:need finish training



def split(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(img_gray, 130, 255, cv2.THRESH_OTSU)
    #cv2.imshow("out", img_thresh)
    h, w = np.shape(img_thresh)
    # print(h, w)
    y_hist = []
    for i in range(w):
        cnt = 0
        for j in range(h):
            if img_thresh[j][i] != 0:
                cnt += 1
        y_hist.append(cnt)
    # print(y_hist)
    x_hist = []
    for i in range(h):
        cnt = 0
        for j in range(w):
            if img_thresh[i][j] != 0:
                cnt += 1
        x_hist.append(cnt)
    #print(x_hist)
    mean = np.argmax(np.bincount(y_hist))
    # print(mean)
    # sliding-window algorithm
    st = []
    en = []
    flag = 0
    for j in range(w):
        if flag == 0 and y_hist[j] - mean > mean:
            st.append(j)
            flag = 1
        if flag == 1 and y_hist[j] - mean <= mean:
            en.append(j)
            flag = 0
    x_st = 0
    x_en = 0
    for i in range(h):
        if x_hist[i] != 0 and x_st == 0:
            x_st = i
        if x_hist[i] == 0 and x_st != 0 and x_en == 0:
            x_en = i
    #print(x_st, x_en)
    #print(st, en)
    size = len(en)
    batch = []
    for i in range(size):
        if en[i] - st[i] < 10:
            continue
        else:
            tempfile = img_thresh[0:h, st[i]:en[i]]
            tempfile = cv2.resize(tempfile, (32, 32), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow(str(i),tempfile)
            # cv2.waitKey(0)
            #rec_number([tempfile.flatten()/255],1)
            batch.append(tempfile.flatten()/255)
    res = rec_number(batch)
    return res

if __name__ == "__main__":
    #data.load_data()



