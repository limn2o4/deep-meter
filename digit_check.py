import tensorflow as tf
import numpy as np
import scipy as sci
import argparse
import data.input_data as data
import random
import time

#version 2 leNet-5

X = tf.placeholder("float",shape=[None,1024])
Y = tf.placeholder("float",shape=[None,10])
keep_prob = tf.placeholder(tf.float32)
def get_network():
    x_img = tf.reshape(X, [-1, 32, 32, 1])
    with tf.variable_scope("l1-conv1"):
        w1 = tf.Variable(tf.random_normal([5, 5, 1, 32], dtype='float'))
        b1 = tf.Variable(tf.random_normal([32]))
        conv1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

    with tf.variable_scope("l2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("l3-conv2"):
        w2 = tf.Variable(tf.random_normal([5, 5, 32, 64], dtype='float'))
        b2 = tf.Variable(tf.random_normal([64]))

        relu2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME'), b2))

    with tf.variable_scope("l4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("l5-fc1"):
        w3 = tf.Variable(tf.random_normal([8 * 8 * 64, 1024]))
        b3 = tf.Variable(tf.random_normal([1024]))
        pool2_vec = tf.reshape(pool2, [-1, w3.get_shape().as_list()[0]])
        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool2_vec, w3),b3))

    with tf.variable_scope("l6-output"):

        fc1 = tf.nn.dropout(fc1, keep_prob)
        w3 = tf.Variable(tf.random_normal([1024, 10]))
        b3 = tf.Variable(tf.random_normal([10]))
        output = tf.nn.softmax(tf.nn.bias_add(tf.matmul(fc1, w3), b3))
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
                batch = data.next_batch_by_num(100,4)
                # print(batch)
                _,_loss = sess.run([train_step,loss],feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})
                if i % 100 == 0:
                    batch_test = data.next_batch_by_num(100,4)
                    pri_result = output.eval(feed_dict={X:batch[0],Y:batch[1],keep_prob:0.5})
                    #loss = sess.run(cross_entropy,feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
                    acc = accuracy.eval(feed_dict={X: batch_test[0], Y: batch_test[1], keep_prob: 1.0})
                    print("step = {} loss = {} acc = {}".format(i, _loss,acc))
                    #saver.save(sess,"./
                    print("result = {} correct = {}".format(pri_result[0],batch_test[1][0]))
    time_en = time.time()
    print("using :{}s".format(time_en - time_st))



if __name__ == "__main__":
    data.load_data()
    #print(data.next_batch(1))
    # for i in range(3) :
    #     data.next_batch(10)
    #batch = data.next_batch_by_num(30,4)
    #print(batch[0][0])
    out = get_network()
    train_network(out)

