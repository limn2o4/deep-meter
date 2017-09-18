import tensorflow as tf
import numpy as nu
import scipy as sci
import tempfile
import cv2
IMAGE_SIZE = 128
IMAGE_PIXELS = 128*128

def create_weights(shape):
    return tf.truncated_normal(shape,stddev=1.0)

def create_biase(shape):
    return tf.constant(0.1,shape=shape)

def create_convLayer(x,w):
    return tf.nn.conv2d(x,w,strides= [1,1,1,1],padding= 'SAME')

def create_poolingLayer(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

def build_CNN(image):


    with tf.name_scope('reshape'):
        x_image = tf.reshape(image,[-1,IMAGE_SIZE,IMAGE_SIZE,1])
    
    with tf.name_scope('conv1'):
        w_conv1 = create_weights([5,5,1,32])
        b_conv1 = create_biase([32])
        hyp_conv1 = tf.nn.relu(create_convLayer(x_image,w_conv1)+b_conv1)
    with tf.name_scope('pool1'):
        hyp_pool1 = create_poolingLayer(hyp_conv1)
    with tf.name_scope('conv2'):
        w_conv2 = create_weights([5,5,32,64])
        b_conv2 = create_biase([64])
        hyp_conv2 = tf.nn.relu(create_convLayer(hyp_pool1,w_conv2),b_conv2)
    with tf.name_scope('pool2'):
        hyp_pool2 = create_poolingLayer(hyp_conv2)
    #full_connect
    with tf.name_scope('fc1'):
        w_fc1 = create_weights([7*7*64,1024])
        b_fc1 = create_biase([1024])

        hyp2_flat = tf.reshape(hyp_pool2,[-1,7*7*64])
        hyp_fc1 = tf.matmul(hyp2_flat,w_fc1)+b_fc1
    
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        hyp_fc1_drop = tf.nn.dropout(hyp_fc1,keep_prob)

    with tf.name_scope('fc2'):
        w_fc2 = create_weights([1024,10])
        b_fc2 = create_biase([10])

        y_fc2= tf.matmul(hyp_fc1_drop,w_fc2)+b_fc2

    return y_fc2,keep_prob


    def main():
        #need a lable function and read function
        data = 0
        x = tf.placeholder(tf.float32,[None,IMAGE_PIXELS])
        y = tf.placeholder(tf.float32,shape = [None,10])

        y_ = tf.placeholder(tf.float32,[None,10])
        y_conv,keep_prob = build_CNN(x)

        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_conv))
        
        with tf.name_scope('AdamOptmizer'):
            trainer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.arg_max(y_conv,1),tf.arg_max(y_,1))
            correct_prediction = tf.cast(correct_prediction,tf.float32)
        accuary = tf.reduce_mean(correct_prediction)

        graph_location = tempfile.mkdtemp()
        print("location : %s".format{graph_location})
        writer = tf.summary.FileWriter(graph_location)
        writer.add_graph(tf.get_default_graph())

        #train process:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(300):
                batch = data.next(25)
                if i%30 == 0:
                    tarin_accuracy = accuary.eval(feed_dict ={x:batch[0],y_:batch[1],keep_prob:1.0})
                    print("step %d accuracy %g".format(i,tarin_accuracy))
                trainer.run(feed_dict = {x:batch[0],y:batch[1],keep_prob = 0.5})
        
        print("test accuacy = %g".format(accuary.eval(feed_dict = {x:data.test_images,y_:data.test.labels,keep_prob = 1.0})))
        
        


