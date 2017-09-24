import tensorflow as tf
import numpy as nu
import scipy as sci
import tempfile
import cv2
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_PIXELS = 128*128



def build_cnn(w_alpha= 0.1,b_alpha = 0.1):
    X = tf.placehloder(tf.float32,shpe = [None])
    Y = tf.placeholder(tf.float32,shape = [None])
    keep_prob = tf.placeholder(tf.float32)
    x = tf.reshape(X,shape= [-1,IMAGE_HEIGHT,IMAGE_WIDTH,1])

    w_c1 = tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
    b_c1 = tf.Variable(b_alpha*tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,[1,1,1,1],padding= 'SAME'),b_c1))
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding= 'SAME')
    conv1 = tf.nn.dropout(conv1,keep_prob)

    w_c2 = tf.Variable(w_alpha*tf.random_normal([3,3,32,64]))
    b_c2 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2 = tf.nn.dropout(conv2,keep_prob)

    w_c3 = tf.Variable(w_alpha*tf.random_normal([3,3,64,64]))
    b_c3 = tf.Variable(b_alpha*tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3))
    conv3 = tf.nn.max_pool(conv3,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3= tf.nn.dropout(conv3,keep_prob)

    w_f = tf.Variable(w_alpha*tf.random_normal([8*20*64,1024]))
    b_f = tf.Variable(b_alpha*tf.random_normal([1024]))
    fc1 = tf.reshape(conv3,[-1,w_f.get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.add(tf.matmul(w_f,conv3),b_f))
    fc1 = tf.nn.dropout(fc1,keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024,4*10]))
    b_out = tf.Variable(b_alpha*tf.random_normal([1024]))
    out = tf.add(tf.matmul(w_out,fc1),b_out)
    out = tf.nn.softmax(out)

    return out
# need more complicated framewrok 3x3conv2d and char2vec


if  __name__ == '__main__':
    netwrok = build_cnn()