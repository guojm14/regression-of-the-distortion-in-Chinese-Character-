# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:41:38 2017

@author: guojm14
"""

from  scipy import io
import os
import threading
import Queue
import numpy as np
from PIL import Image
import random
def noise_gaussion(img,sigma=5):
    
    if random.randint(0,1)==0:
        h=img.shape[0]
        w=img.shape[1]
        for i in xrange(h):
            for j in xrange(w):
                noise=np.random.normal(0,sigma)
                if len(img.shape)==3:
                    img[i,j,:]=noise+img[i,j,:]
                else:
                    img[i,j]=noise+img[i,j]
        return np.clip(img,0,255)
    
    else:
        return img

def noise_salt(img,rate=0.01):
    if random.randint(0,1)==0:
        h=img.shape[0]
        w=img.shape[1]
        m=int(h*w*rate)
        for a in xrange(m):
            j=int(np.random.random()*h)
            i=int(np.random.random()*w)
            if len(img.shape)==3:
                img[j,i,:]=255
            else:
                img[j,i]=255
            j=int(np.random.random()*h)
            i=int(np.random.random()*w)
            if len(img.shape)==3:
                img[j,i,:]=0
            else:
                img[j,i]=0
        return np.clip(img,0,255)
    else:
        return img

class dataloader(threading.Thread):
    def __init__(self,datapath,datalistfile,size=[128,32],batchsize=64,t_name='dataloader',addnoise=False):
        threading.Thread.__init__(self, name=t_name)  
        self.datapath=datapath
        self.datalist= open(datalistfile).readlines()
        self.dataqueue=Queue.Queue(maxsize=10)
        self.bs=batchsize
        self.on=True
        self.index=0
        self.length=len(self.datalist)
        self.epoch=0
        self.size=size
        self.addnoise=addnoise
        random.shuffle(self.datalist)
        print 'inited'
    def run(self):
        while(self.on):
            
            data=[]
            label=[]
            for i in xrange(self.bs):
                dataline=self.datalist[self.index].strip().split()
                imgname=dataline[0]
                img=np.array(Image.open(os.path.join(self.datapath,imgname)).resize(self.size),dtype='float')
                if self.addnoise:
                    img=noise_gaussion(noise_salt(img))
                imlabel=np.array(map(float,dataline[1:]))
                data.append(img)
                label.append(imlabel)
                self.index+=1
                if self.index==self.length:
                    self.index=0
                    self.epoch+=1
                    random.shuffle(self.datalist)
            label=np.array(label)
            data=np.array(data)
            item=(data,label)
            self.dataqueue.put(item)
    def getdata(self):
        return self.dataqueue.get()
    def close(self):
        self.on=False
def testcode():
    a=dataloader('../textgenerator-/transresult/','train.txt')
    a.start()
    data,label= a.getdata()
    print data.shape
    print label.shape
    a.close()

