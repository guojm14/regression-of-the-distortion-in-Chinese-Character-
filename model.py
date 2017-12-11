from __future__ import division
import os
import time
import numpy as np
import tensorflow as tf
from loaddata import dataloader
slim=tf.contrib.slim
class regression(object):
    def __init__(self,sess,batch_size=64,
                num_epoch =25,
                lr=0.001,
                imagesize=[32,128],
                pointnum=[5,2],
                datapath='../textgenerator-/transresult/',
                trainlist='train.txt',
                testlist='test.txt')
        self.sess=sess
        self.lr=lr
        self.batch_size=batch_size
        self.num_epoch=num_epoch
        self.pointnum=[5,2]
        self.trainloader=dataloader(datapath,trainlist)
        self.testloader=dataloader(datapath,testlist)
        self.trainloader.start()
        self.testloader.start()
        
    def build_model(self):
        self.img=tf.placeholder(tf.float32,[self.batch_size,self.imagesize[0],self.imagesize[1],3])
        self.label=tf.placeholder(tf.float32,[self.batch_size,self.pointnum[0]*self.pointnum[1]*2])
        self.labelp=self.local(img)
        self.loss=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(labelp-label),1)))
        self.loss_sum=tf.summary.scalar('loss',self.loss)
    def train(self):
        optimi=tf.train.AdamOptimizer(lr).minimize(self.loss)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        self.loss_sum=tf.summary.merge([self.loss])
        count=0
        while 1:
            
    def local(self,x,is_training=True):
        with  tf.variable_scope('local'):
            with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}):
                conv1=slim.conv2d(x, 64, 3, 1)
                pool1=slim.max_pool2d(conv1, [2, 2])
                conv2=slim.conv2d(pool1, 128, 3, 1)
                pool2=slim.max_pool2d(conv2, [2, 2])
                conv3=slim.conv2d(pool2, 256, 3, 1)
                pool3=slim.max_pool2d(conv3, [2, 2])
                conv4=slim.conv2d(pool3, 512, 3, 1)
                pool4=slim.max_pool2d(conv4, [2, 2])
                temp=slim.flatten(pool4, scope='flatten')
                fc1=tf.nn.relu(slim.fully_connected(inputs=temp, num_outputs=1024, scope='fc1'))
                fc2=tf.tanh(slim.fully_connected(inputs=fc1, num_outputs=self.pointnum[0]*self.pointnum[1]*2, scope='fc2'))
        return fc2
        
