import tensorflow as tf
import numpy as np
import scipy as sci
import argparse
import data.input_data as data
import random


#version 2 leNet-5

X = tf.placeholder("float",shape=[None,1024])
Y = tf.placeholder("float",shape=[None,11])
keep_prob = tf.placeholder(tf.float32)
def get_network():
    x_img = tf.reshape(X, [-1, 32, 32, 1])
    with tf.variable_scope("l1-conv1"):
        w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], dtype='float'))
        b1 = tf.Variable(tf.truncated_normal([32]))

        conv1 = tf.nn.conv2d(x_img, w1, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

    with tf.variable_scope("l2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("l3-conv2"):
        w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], dtype='float'))
        b2 = tf.Variable(tf.truncated_normal([64]))

        relu2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding='SAME'), b2))

    with tf.variable_scope("l4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("l5-fc1"):
        w3 = tf.Variable(tf.truncated_normal([8 * 8 * 64, 1024]))
        b3 = tf.Variable(tf.truncated_normal([1024]))
        pool2_vec = tf.reshape(pool2, [-1, 8 * 8 * 64])
        fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pool2_vec, w3),b3))

    with tf.variable_scope("l6-output"):

        fc1 = tf.nn.dropout(fc1, keep_prob)
        w3 = tf.Variable(tf.truncated_normal([1024, 11]))
        b3 = tf.Variable(tf.truncated_normal([11]))
        output = tf.nn.softmax(tf.nn.bias_add(tf.matmul(fc1, w3), b3))

    return output
def train_network(output):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    prediction = tf.equal(tf.argmax(output,1),tf.argmax(Y,1))

    accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

    with tf.Session() as sess :
        with tf.device("/gpu:0"):
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                batch = data.next_batch(100)
                # print(batch)
                if i % 100 == 0:
                    acc = accuracy.eval(feed_dict={X: batch[0], Y: batch[1], keep_prob: 1.0})
                    print("step = {} arruracy = {}".format(i, acc))
                train_step.run(feed_dict={X: batch[0], Y: batch[1], keep_prob: 0.5})


if __name__ == "__main__":
    data.load_data()
    out = get_network()
    train_network(out)





