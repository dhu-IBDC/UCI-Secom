# This Python file uses the following encoding: utf-8

import numpy as np
import tensorflow as tf
from skimage.io import imsave
import scipy
import matplotlib.pylab as plt


class GAN:
    def __init__(self, data, max_epoch=200, batch_size=256, z_size=128, g_depths=[128, 128, 100],
                 d_depths=[100, 512, 128, 1],
                 keep_prob=0.7):
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.z_size = z_size
        self.z = tf.random_uniform([self.batch_size, self.z_size], minval=-1, maxval=1)
        self.g_depths = g_depths
        self.d_depths = d_depths
        self.keep_prob = keep_prob
        self.data = data

    def build_generator(self, z):
        print (z)

        w1 = tf.Variable(tf.truncated_normal([self.z_size, self.g_depths[0]], stddev=0.1), name="g_w1",
                         dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([self.g_depths[0]]), name="g_b1", dtype=tf.float32)
        h1 = tf.nn.relu(tf.matmul(z, w1) + b1)
        w2 = tf.Variable(tf.truncated_normal([self.g_depths[0], self.g_depths[1]], stddev=0.1), name="g_w2",
                         dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([self.g_depths[1]]), name="g_b2", dtype=tf.float32)
        h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
        w3 = tf.Variable(tf.truncated_normal([self.g_depths[1], self.g_depths[2]], stddev=0.1), name="g_w3",
                         dtype=tf.float32)
        b3 = tf.Variable(tf.zeros([self.g_depths[2]]), name="g_b3", dtype=tf.float32)
        h3 = tf.matmul(h2, w3) + b3
        x_generate = tf.nn.tanh(h3)
        g_params = [w1, b1, w2, b2, w3, b3]
        return x_generate, g_params

    def build_discriminator(self, x_data, x_generated, keep_prob=None):
        if keep_prob is None:
            keep_prob = self.keep_prob
        x_in = tf.concat([x_data, x_generated], 0)
        w1 = tf.Variable(tf.truncated_normal([self.d_depths[0], self.d_depths[1]], stddev=0.1), name="d_w1",
                         dtype=tf.float32)
        b1 = tf.Variable(tf.zeros([self.d_depths[1]]), name="d_b1", dtype=tf.float32)
        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
        w2 = tf.Variable(tf.truncated_normal([self.d_depths[1], self.d_depths[2]], stddev=0.1), name="d_w2",
                         dtype=tf.float32)
        b2 = tf.Variable(tf.zeros([self.d_depths[2]]), name="d_b2", dtype=tf.float32)
        h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
        w3 = tf.Variable(tf.truncated_normal([self.d_depths[2], 1], stddev=0.1), name="d_w3", dtype=tf.float32)
        b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
        h3 = tf.matmul(h2, w3) + b3
        y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [self.batch_size, -1], name=None))
        y_generated = tf.nn.sigmoid(tf.slice(h3, [self.batch_size, 0], [-1, -1], name=None))
        d_params = [w1, b1, w2, b2, w3, b3]
        return y_data, y_generated, d_params

    def show_result(self, data):
        print(data)

    def drawing(x):
        plt.figure(num=1)
        plt.plot(x, color='g', linewidth=0.8)
        plt.title('prob')
        plt.grid()
        plt.legend(['loss'])
        plt.pause(0.0001)

    def gan_data(self):
        X = tf.placeholder("float", shape=[None, 100])
        Z = tf.placeholder("float", shape=[None, 128])
        zzz = np.random.uniform(0.0, 1.0, size=[self.batch_size, 128])

        X_generated, g_paramters = self.build_generator(z=Z)
        y_data, y_generated, d_paramters = self.build_discriminator(X, X_generated)

        d_loss = tf.reduce_mean(- (tf.log(y_data) + tf.log(1 - y_generated)))
        g_loss = tf.reduce_mean(- tf.log(y_generated))

        d_optimizer = tf.train.AdamOptimizer(0.00007)
        g_optimizer = tf.train.AdamOptimizer(0.0001)

        d_trainer = d_optimizer.minimize(d_loss, var_list=d_paramters)
        g_trainer = g_optimizer.minimize(g_loss, var_list=g_paramters)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        self.data -= 1
        self.data *= 2

        for i in range(self.max_epoch):
            sess.run(d_trainer, feed_dict={X: self.data, Z: zzz})
            sess.run(g_trainer, feed_dict={X: self.data, Z: zzz})
            if i % 1 == 0:
                y_data_prob = sess.run(y_data, feed_dict={X: self.data, Z: zzz})
                y_generate_prob = sess.run(y_generated, feed_dict={X: self.data, Z: zzz})
                plt.figure(num=1)
                plt.plot(y_data_prob, color='g', linewidth=0.8)
                plt.plot(y_generate_prob, color='y', linewidth=0.8)
                plt.title('discriminator prob :current training time: %d' % i)
                plt.grid()
                plt.legend(['discriminator prob'])
                plt.ylim(0, 1)
                plt.pause(0.0001)

            gan_data = sess.run(X_generated, feed_dict={Z: np.random.uniform(0, 1, size=[768, 128])})
        return gan_data
