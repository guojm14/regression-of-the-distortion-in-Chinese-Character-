from __future__ import division
import os
import time
import numpy as np
import tensorflow as tf
from loaddata import dataloader
from TPS_STN import TPS_STN
from utils import *
slim=tf.contrib.slim
lr=0.001
class regression(object):
    def __init__(self,sess,batch_size=64,
                num_epoch =25,
                lr=0.001,
                imagesize=[32,128],
                pointnum=[5,2],
                datapath='../textgenerator-/transresult_v1/',
                trainlist='../textgenerator-/list.txt',
                testlist='testlist.txt'):
        self.sess=sess
        self.lr=lr
        self.batch_size=batch_size
        self.num_epoch=num_epoch
        self.pointnum=[5,2]
        self.imagesize=imagesize
        self.jobname='addnoise'
        if not os.path.exists(self.jobname):
            os.mkdir(self.jobname)
        self.sampledir=os.path.join(self.jobname,'sample/')
        if not os.path.exists(self.sampledir):
            os.mkdir(self.sampledir)
        self.modeldir=os.path.join(self.jobname,'checkpoint/')
        if not os.path.exists(self.modeldir):
            os.mkdir(self.modeldir)
        self.logdir=os.path.join(self.jobname,'log/')
        if not os.path.exists(self.logdir):
            os.mkdir(self.logdir)
       
        self.trainloader=dataloader(datapath,trainlist,batchsize=self.batch_size,t_name='train',addnoise=True)
        self.testloader=dataloader('test/',testlist,batchsize=64,t_name='test')
        self.trainloader.start()
        self.testloader.start()
        self.build_model()
        self.saver=tf.train.Saver()
    def build_model(self):
        print 'building model .......'
        self.img=tf.placeholder(tf.float32,[self.batch_size,self.imagesize[0],self.imagesize[1],3])
        self.label=tf.placeholder(tf.float32,[self.batch_size,self.pointnum[0]*self.pointnum[1]*2])
        self.labelp=self.local(self.img)
        self.loss=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.labelp-self.label),1)))
        self.loss_sum=tf.summary.scalar('loss',self.loss)
        self.img_re=TPS_STN(self.img,self.pointnum[0],self.pointnum[1],tf.reshape(self.labelp,[-1,5*2,2]),self.imagesize+[3])
        print 'built'
    def train(self):
        optimi=tf.train.AdamOptimizer(lr).minimize(self.loss)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        self.loss_sum1=tf.summary.merge([self.loss_sum])
        self.writer=tf.summary.FileWriter(self.logdir,self.sess.graph)
        

        print 'start'        
        for i in xrange(int(self.trainloader.length/self.batch_size*self.num_epoch)):
            
            traindata,trainlabel=self.trainloader.getdata()
            if i%100==1:
                _,loss_str,loss=sess.run([optimi,self.loss_sum1,self.loss],feed_dict={self.img:traindata,self.label:trainlabel})
                self.writer.add_summary(loss_str,i)
                print 'epoch '+str(self.trainloader.epoch)+' iter '+str(i)+' loss '+str(loss)
            else:
                _=sess.run([optimi],feed_dict={self.img:traindata,self.label:trainlabel})
            if i%1000==1:
                
                self.saver.save(self.sess,self.modeldir,global_step=i)
                testdata,testlabel=self.testloader.getdata()
                testdata=testdata.reshape([64,32,128,1]).repeat(3,3)
                loss,reimg,plabel=sess.run([self.loss,self.img_re,self.labelp],feed_dict={self.img:testdata,self.label:testlabel})
                print 'test '+str(i)+'loss '+str(loss)
                print reimg.shape
                save_images(reimg,[8,8],self.sampledir+str(i)+'reimgtest.jpg')
                save_images_point(testdata,[8,8],self.sampledir+str(i)+'imgtest.jpg',plabel)
                testdata,testlabel=self.trainloader.getdata()
                
                loss,reimg,plabel=sess.run([self.loss,self.img_re,self.labelp],feed_dict={self.img:testdata,self.label:testlabel})
                print 'trainsample '+str(i)+'loss '+str(loss)
                print reimg.shape
                save_images(reimg,[8,8],self.sampledir+str(i)+'reimg.jpg')
                    
                save_images_point(testdata,[8,8],self.sampledir+str(i)+'img.jpg',plabel)   
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
                fc2=fcforpoint(fc1,self.pointnum[0],self.pointnum[1])
        return fc2
def fcforpoint(input_, nx=10,ny=2):

    shape = input_.get_shape().as_list()
    top=np.concatenate([np.expand_dims(np.linspace(-1,1,nx),1),np.full([nx,1],1)],1)
    bot=np.concatenate([np.expand_dims(np.linspace(-1,1,nx),1),np.full([nx,1],-1)],1)
    v=np.concatenate([bot,top],0)
    v=v.reshape(nx*ny*2)
    with tf.variable_scope("pointLinear"):
        matrix = tf.get_variable("Matrix", [shape[1], nx*ny*2], tf.float32,
                 tf.constant_initializer(0.0))
        bias = tf.get_variable("bias", [nx*ny*2],
      initializer=tf.constant_initializer(v))
    return tf.tanh((tf.matmul(input_, matrix) + bias))
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
with tf.Session(config=run_config) as sess:
    a=regression(sess)

    a.train()        
