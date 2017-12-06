import tensorflow as tf
import numpy as np
import scipy as sci
import argparse
import data.input_data as data
import random
import time
import cv2
#version 2 leNet-5

X = tf.placeholder("float",shape=[None,1024])
Y = tf.placeholder("float",shape=[None,10])
keep_prob = tf.placeholder(tf.float32)
def get_network():
    x_img = tf.reshape(X, [-1, 32, 32, 1])
    with tf.variable_scope("l1-conv1"):
        w1 = tf.get_variable("weights",[5,5,1,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
        relu1 = tf.nn.dropout(relu1,keep_prob)

    with tf.variable_scope("l2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("l3-conv2"):
        w2 = tf.get_variable("weights",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable("bias",[64],initializer=tf.constant_initializer(0.0))
        relu2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME'), b2))
        relu2 = tf.nn.dropout(relu2,keep_prob)
    with tf.variable_scope("l4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("l5-fc1"):
        w3 = tf.get_variable("weights",[8*8*64,1024],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b3 = tf.get_variable("bias",[1024],initializer=tf.constant_initializer(0.0))
        pool2_vec = tf.reshape(pool2, [-1, w3.get_shape().as_list()[0]])
        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool2_vec, w3),b3))
        fc1 = tf.nn.dropout(fc1,keep_prob)

    with tf.variable_scope("l6-output"):

        fc1 = tf.nn.dropout(fc1, keep_prob)
        w3 = tf.get_variable("weights",[1024,10],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b3 = tf.get_variable("bias",[10],initializer=tf.constant_initializer(0.0))
        #output = tf.nn.softmax(tf.nn.bias_add(tf.matmul(fc1, w3), b3))
        output = tf.matmul(fc1,w3)+b3
    return output
def train_network(output):

    globe_step = tf.Variable(0,)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))

    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    #pred = tf.reshape(output,[-1,10])
    max_p = tf.argmax(tf.reshape(output,[-1,10]),1)
    max_l = tf.argmax(tf.reshape(Y,[-1,10]),1)
    prediction = tf.equal(max_l,max_p)

    accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

    saver = tf.train.Saver()
    time_st = time.time()
    with tf.Session() as sess :
        #saver.restore(sess,tf.train.latest_checkpoint('.'))
        with tf.device("/gpu:0"):
            sess.run(tf.global_variables_initializer())
            for i in range(2000):
                batch = data.next_batch(100)
                #print(batch)
                _,_loss = sess.run([train_step,loss],feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.75})
                if i % 100 == 0:
                    batch_test = data.next_batch(100)
                    #pri_result = output.eval(feed_dict={X:batch[0],Y:batch[1],keep_prob:0.5})
                    #loss = sess.run(cross_entropy,feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
                    acc = accuracy.eval(feed_dict={X: batch_test[0], Y: batch_test[1], keep_prob: 1.0})
                    print("step = {} loss = {} acc = {}".format(i, _loss,acc))
                    # if(i == 1000) :
                    #     saver.save(sess,"./m.model",global_step=i)
                    #print("result = {} correct = {}".format(pri_result[0],batch_test[1][0]))
    time_en = time.time()
    print("using :{}s".format(time_en - time_st))


def rec_number(image):
    num = len(image)
    output = get_network()
    saver = tf.train.Saver()
    st = time.time()
    with tf.Session() as sess ,tf.device("/gpu:0"):

        saver.restore(sess,tf.train.latest_checkpoint("."))
            #print(tf.train.latest_checkpoint("."))
            #image_input = image.flatten()/255
        pri_result = sess.run(output,feed_dict={X:image,keep_prob:1.0})
            #print(result)
        en = time.time()
        res = []
        for i in range(num):
            res.append(np.argmax(pri_result[i]))

    print("Total :{},used {}s,result = {}".format(num,en - st,res))
    return res
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
    data.load_data()

    parser = argparse.ArgumentParser()
    parser.add_argument('-Input',type = str)
    parser.add_argument('-Output',type = str)


    args = parser.parse_args()
    input_path = args.Input
    output_path = args.Output

    image = cv2.imread(input,0)
    with open(output_path,'a') as file:
        result = split(image)
        file.write("result:{}".format(result))
    # input_path,output_path
    #out = get_network()
    #train_network(get_network())

    # image = cv2.imread("./1.jpg")
    # cv2.imshow("orginal",image)
    # cv2.waitKey(0)
    # split(image)
    #testing
    # image = cv2.imread("./data/image/"+str(random.randint(1,30000))+".jpg",0)
    # cv2.imshow("out1",image)
    # cv2.waitKey(0)
    # rec_number([image.flatten()/255],1)



