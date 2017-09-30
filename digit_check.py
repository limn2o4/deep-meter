import tensorflow as tf
import numpy as np
import scipy as sci
import tempfile
import argparse
import data.input_data as data
import cv2
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
IMAGE_PIXELS = 160*60
CHARSET_LEN = 10

def get_next_batch(batch_size = 10):
    batch_x = np.zeros([batch_size,IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size,CHARSET_LEN])
    for i in range(1,batch_size):
        test,image = data.get_data(i)
        #print(image)
        #np.reshape(image,[IMAGE_HEIGHT,IMAGE_WIDTH])
        batch_x[i:] = image.flatten()/255
        batch_y[i:] = test
            #print(batch_y[i:])
    return batch_x,batch_y

def text2vec(text):
    vector = np.zeros(200)
    for i,c in enumerate(text):
        idx = i*10+ord(c) - ord('0')
        vector[idx] = 1
    return vector

def vec2text(vector):
    text = np.zeros(CHARSET_LEN)
    #vec = np.nonzero(vector)[0]
    for i in range(len(vector)):
        if vector[i] == 1:
            idx = i // 10
            c = i % 10
            #print(idx,c)
            text[idx] = c
    return  text
X = tf.placeholder(tf.float32,shape = [None])
Y = tf.placeholder(tf.float32,shape = [None])
keep_prob = tf.placeholder(tf.float32)
def build_cnn(w_alpha= 0.1,b_alpha = 0.1):


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
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1,w_f),b_f))
    fc1 = tf.nn.dropout(fc1,keep_prob)

    w_out = tf.Variable(w_alpha*tf.random_normal([1024,4*10]))
    b_out = tf.Variable(b_alpha*tf.random_normal([1024]))
    out = tf.add(tf.matmul(w_out,fc1),b_out)
    out = tf.nn.softmax(out)

    return out
# need more complicated framewrok 3x3conv2d and char2vec
def train_cnn():
    output = build_cnn()
    #loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    hyp = tf.reshape(output , [-1,10,4])
    max_p = tf.argmax(hyp,2)
    max_l = tf.argmax(tf.reshape(Y,[-1,10,4],),2)
    corrected = tf.equal(max_l,max_p)
    accuracy = tf.reduce_mean((tf.cast(corrected,tf.float32)))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(100):
            batch_x,batch_y = get_next_batch(10)
            _,loss_ = sess.run([optimizer,loss],feed_dict= {X:batch_x,Y:batch_y})
            print(step,loss_)
            if step % 10 == 0:
                accuracy_ = sess.run(accuracy,feed_dict={X:batch_x,Y:batch_y})
    saver.save(sess,"digital_rec.model",global_step=step)

def rec_by_cnn(image):
    out = build_cnn();
    saver = tf.train.Saver();
    with tf.Session() as sess:
        saver.restore(sess,tf.train.latest_checkpoint('.'))

        predict = tf.argmax(tf.reshape(out,[-1,12,10,]),2)
        text_list = sess.run(predict,feed_dict={X:[image],keep_prob:1})
        test = text_list[0].tolist()
        vector = np.zeros(12*10)
        i = 0
        for p in test:
            vector[i*10+p] = 1
            i+=1
        result = vec2text(vector)
    print(predict)

if  __name__ == '__main__':
    #netwrok = build_cnn()
    train_cnn()
    #file = open(r'D:\project\deep-meter\data\numbers.txt',"r")
    #get_next_batch()