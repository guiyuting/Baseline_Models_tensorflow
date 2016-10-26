'''
vanilla cnn in mnist
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import logging
import dataprepare
import random
logging.basicConfig(filename='cnn_mnist.log',level=logging.INFO)
'''
def dataprepare():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    return mnist.train, mnist.validation, mnist.test
'''
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = "SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], \
                strides = [1,2,2,1], padding = "SAME")
class cnn():


    def __init__(self):
        # hyparameter
        self.input_size    = 784
        self.lr            = 0.001
        self.batch_size    = 100
        self.keep_prob     = 0.6
        self.epochs        = 10


        # input
        self.image = tf.placeholder(tf.float32, [None, self.input_size])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.x = tf.reshape(self.image, [-1,28,28,1])
        # forward path 1-64, 64-128
        W_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([5,5,32,128])
        b_conv2 = bias_variable([128])
        W_conv3 = weight_variable([3,3,128,256])
        b_conv3 = bias_variable([256])
        W_conv4 = weight_variable([3,3,256,256])
        b_conv4 = bias_variable([256])

        W_fc1   = weight_variable([2*2*256, 256])
        b_fc1   = bias_variable([256])
        W_fc2   = weight_variable([256, 10])
        b_fc2   = bias_variable([10])
        
        h_conv1 = tf.nn.dropout(tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1),\
                  keep_prob = self.keep_prob)
        h_pool1 = max_pool_2x2(h_conv1) #
        h_conv2 = tf.nn.dropout(tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2),\
                  keep_prob = self.keep_prob)
        h_pool2 = max_pool_2x2(h_conv2) #8*8*128
        h_conv3 = tf.nn.dropout(tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3),\
                  keep_prob = self.keep_prob)
        h_pool3 = max_pool_2x2(h_conv3) #4*4*256 ceil(7/2)
        
        h_conv4 = tf.nn.dropout(tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4),\
                  keep_prob = self.keep_prob)
        h_pool4 = max_pool_2x2(h_conv4) #2*2*256 
        h_pool4_flat = tf.reshape(h_pool4, [-1, 2*2*256])
        h_fc1   = tf.nn.dropout(tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + \
                  b_fc1), keep_prob = self.keep_prob)
        h_fc2   = tf.matmul(h_fc1, W_fc2) + b_fc2 

        correct_prediction = tf.equal(tf.argmax(h_fc2,1), tf.argmax(self.y,1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                h_fc2, self.y))
        self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.8).minimize(self.loss)
        # Alternative: GradientDescentOptimizer
        
        self.testy = tf.placeholder(tf.float32, [None,10])
        correct_test_pred = tf.equal(tf.argmax(h_fc2,1), tf.argmax(self.testy,1))
        self.testacc = tf.reduce_mean(tf.cast(correct_test_pred, tf.float32))

        self.init = tf.initialize_all_variables()
        self.sess = tf.InteractiveSession()

    def train_mnist(self, training_set, valid):
        print("Training")
        self.sess.run(self.init)
        for step in xrange(100000):
            batch = training_set.next_batch(self.batch_size)
            feed = {self.image: batch[0], self.y: batch[1]}
            _, loss, acc = self.sess.run([self.optimizer, self.loss, self.acc],\
                         feed_dict = feed)
            if step % 2 == 0:
                print("Step: %s, Cost: %s, Acc: %.3f" %(step, loss, acc))
                logging.info("Step: %s, Cost: %s, Acc: %.3f" %(step, loss, acc))
            if step % 100 == 0 and step != 0:
                testacc = self.sess.run([self.testacc],\
                feed_dict = {self.image: valid.images, self.testy:valid.labels}) 
                print("Valid, Step: %s, Acc: %.3f" %(step, acc))
                loggint.info("Step: %s, Cost: %s, Acc: %.3f" %(step, loss, acc))

    def train_cifar(self,train, valid):
        print("Training")
        self.sess.run(self.init)
        total_batch = int(train['images'].shape[0] / self.batch_size)

        for e in xrange(self.epochs):
            for i in xrange(total_batch):
                onehoty = self.sess.run(tf.one_hot(\
                        train["labels"][self.batch_size*i:\
                        self.batch_size*(i+1)], depth = 10, \
                        on_value = 1.0, off_value = 0.0))
                feed={self.image:train['images'][self.batch_size*i : \
                      self.batch_size*(i+1)],
                      self.y:onehoty}
                _, loss, acc = self.sess.run([self.optimizer, self.loss, \
                               self.acc], feed_dict = feed)
                if i % 2 == 0:
                    print("Step: %s, Cost: %s, Acc: %f.3" \
                         %(self.epochs*e+self.batch_size*i, loss, acc))
                    logging.info("Step: %s, Cost: %s, Acc: %f.3" \
                         %(self.epochs*e+self.batch_size*i, loss, acc))
def main():

    training, valid, test = dataprepare.read_mnist()
    mnist_cnn = cnn()
    mnist_cnn.train_mnist(training, valid)

if __name__ == "__main__":
    main()








