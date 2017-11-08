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
            for i in range(1000):
                batch = data.next_batch(100)
                #print(batch)
                _,_loss = sess.run([train_step,loss],feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.75})
                if i % 100 == 0:
                    batch_test = data.next_batch(100)
                    #pri_result = output.eval(feed_dict={X:batch[0],Y:batch[1],keep_prob:0.5})
                    #loss = sess.run(cross_entropy,feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
                    acc = accuracy.eval(feed_dict={X: batch_test[0], Y: batch_test[1], keep_prob: 1.0})
                    print("step = {} loss = {} acc = {}".format(i, _loss,acc))
                    if(i == 1000) :
                        saver.save(sess,"./m.model",global_step=i)
                    #print("result = {} correct = {}".format(pri_result[0],batch_test[1][0]))
    time_en = time.time()
    print("using :{}s".format(time_en - time_st))


def rec_number(image):
    output = get_network()
    saver = tf.train.Saver()
    with tf.Session() as sess ,tf.device("/gpu:0"):
        saver.restore(sess,tf.train.latest_checkpoint("."))
        #print(tf.train.latest_checkpoint("."))
        #image_input = image.flatten()/255
        result = sess.run(output,feed_dict={X:[image],keep_prob:1.0})
        print(result)
        print(np.argmax(result))

if __name__ == "__main__":
    data.load_data()
    #print(data.next_batch(1))
    # for i in range(3) :
    #     data.next_batch(10)
    #batch = data.next_batch_by_num(30,4)
    #print(batch[0][0])
    #out = get_network()
    #train_network(get_network())
    image = cv2.imread("./data/image/10.jpg",0)
    image_in = image.flatten()/255
    rec_number(image_in)


