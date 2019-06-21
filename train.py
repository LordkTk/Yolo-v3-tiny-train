# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:23:13 2019

@author: cfd_Liu
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import random
import time
from train_utils import *
from prediction import post_process, decode
from mAP import get_mAP
tf.reset_default_graph()

class_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
               'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
               'sofa', 'train', 'tvmonitor']

def build_net(x, is_training, trainable=False):
    net, route2 = build_darknet(x, is_training, trainable=trainable)
    out = build_detnet(net, route2, is_training)
    return out
def load_img(path, batchList, imgSize):
    imgbatch = []
    for filename in batchList:
        filepath = os.path.join(path, filename)
        img = cv2.resize(cv2.imread(filepath, 1), (imgSize, imgSize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[np.newaxis,:,:,:].astype(np.float32) / 255
        imgbatch.append(img)
    imgbatch = np.concatenate(imgbatch, axis=0)
    return imgbatch
def train_yolo(warm_up=False, fst_stage=False, scd_stage=False, resume=False, predict=False):
    imgInfo = np.load('./data/dataset.npy').item()
    trainList = np.load('./data/trainList.npy').item()[1]
    testList = np.load('./data/testList.npy').item()[1]
#    trainList = testList + trainList  #use all imgs for training1
    testLossList = testList[:10]
    path = np.load('./data/path.npy').item()[1]
    
    imgSize = 416  
    anchors = np.array([35.3655,52.1796, 89.6969,130.258, 111.745,236.696, 295.59,190.8, 185.47,341.823, 358.706,361.342]).reshape([6, 2]).astype(np.float32)
    anchorCoor = proc_anchors(anchors, imgSize)
    
    x = tf.placeholder(tf.float32, [None, imgSize, imgSize, 3])
    y1 = tf.placeholder(tf.float32, [None, 13, 13, 3, 25])
    y2 = tf.placeholder(tf.float32, [None, 26, 26, 3, 25])
    bbox = tf.placeholder(tf.float32, [None, None, 4])
    
    is_training = tf.placeholder(tf.bool)
    
    global_step = tf.Variable(0, name='global_step')
    lr_init = 5e-4
    
    sess = tf.Session()
    exp_name = 'exp0'
    if warm_up:
        out = build_net(x, is_training)
        epochMax = 2
        batchSize = 20
        decay_step = int(len(trainList)/batchSize * epochMax)
        save_name = 'warm_up'
        
        learn_rate = tf.train.polynomial_decay(1e-10, global_step, decay_step, lr_init)
        loss = cal_loss(out, [y1, y2], anchorCoor, bbox, focal_loss=True)
        
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss[0], global_step)
        
        sess.run(tf.global_variables_initializer())
        v_list = [v for v in tf.global_variables() 
        if 'Adam' not in v.name and 'beta1_power' not in v.name and 'beta2_power' not in v.name and 'det1_3' not in v.name and 'det2_3' not in v.name and 'global_step' not in v.name]
        saver = tf.train.Saver(v_list)
        saver.restore(sess, './Weights/weightsInit.ckpt')
    elif fst_stage:
        out = build_net(x, is_training)
        epochMax = 4
        batchSize = 20
        decay_step = int(len(trainList)//batchSize + 1)
        decay_rate = 0.1**(1/epochMax)
        save_name = 'fst_stage'
        
        learn_rate = tf.train.exponential_decay(lr_init, global_step, decay_step, decay_rate, staircase=True)
        loss = cal_loss(out, [y1, y2], anchorCoor, bbox, focal_loss=True)
        
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss[0], global_step)
        
        if resume:
            saver = tf.train.Saver()
            file = tf.train.latest_checkpoint('./Weights/%s/fst_stage/' %exp_name)
            saver.restore(sess, file)
        else:
            sess.run(tf.global_variables_initializer())
            v_list = [v for v in tf.global_variables() if 'global' not in v.name]
            saver = tf.train.Saver(v_list)
            file = tf.train.latest_checkpoint('./Weights/%s/warm_up/' %exp_name)
            saver.restore(sess, file)
    elif scd_stage:
        out = build_net(x, is_training, trainable=True)
        epochMax = 20
        batchSize = 8
        decay_step = int(len(trainList)//batchSize + 1)
        decay_rate = 0.1**(1/epochMax)
        save_name = 'scd_stage'
        
        learn_rate = tf.train.exponential_decay(lr_init/10, global_step, decay_step, decay_rate, staircase=True)
        loss = cal_loss(out, [y1, y2], anchorCoor, bbox, focal_loss=True, regularization=True)
        
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_op):
            train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss[0], global_step)
        
        if resume:
            saver = tf.train.Saver()
            file = tf.train.latest_checkpoint('./Weights/%s/scd_stage/' %exp_name)
            saver.restore(sess, file)
        else:
            sess.run(tf.global_variables_initializer())
            v_list = [v for v in tf.global_variables() 
            if 'Adam' not in v.name and 'beta1_power' not in v.name and 'beta2_power' not in v.name and 'global' not in v.name]
            saver = tf.train.Saver(v_list)
            file = tf.train.latest_checkpoint('./Weights/%s/fst_stage/' %exp_name)
            saver.restore(sess, file)
    if predict:
        out = build_net(x, is_training)
        v_list = [v for v in tf.global_variables() 
        if 'Adam' not in v.name and 'beta1_power' not in v.name and 'beta2_power' not in v.name]
        saver = tf.train.Saver(v_list)
        file = tf.train.latest_checkpoint('./Weights/%s/scd_stage/' %exp_name)
        saver.restore(sess, file)
        
        pathList = []
        for test in testList[:10]:
            pathList.append(os.path.join(path, test))
        pathList = ['C:/Users/cfd_Liu/Desktop/Machine Learning/Code/PracticeCode/TensorFlow Learning/OpenCV/LaneDet/YOLO/img/sample_person.jpg']
        post_process(sess, x, is_training, out, pathList, imgSize, 20, class_names, anchors)
    else:
        'Put all tf operations out of loop! Or the tf computing graphs will add consistently'
        decode_op = decode(out, imgSize, 20, anchors)
        'Confirm the tf graphs have been built before loop with this function. But it would conflict with Saver()'
#        sess.graph.finalize()
        
        testLoss = []
        mAP0 = 0
        for epoch in range(epochMax):
            timeS = time.time()
            sampleList = trainList.copy()
            step = 0
            save = False
            
            while sampleList != []:
                batchList = []
                if len(sampleList) > batchSize:
                    randint = random.sample(range(0, len(sampleList)), batchSize)
                else:
                    randint = list(range(len(sampleList)))
                for i in randint:
                    batchList.append(sampleList[i])
                for i in range(len(randint)):
                    sampleList.remove(batchList[i])
                
                imgbatch = load_img(path, batchList, imgSize)
                
                y, bbox_gr = cal_y(batchList, imgInfo, anchorCoor)
                
                feed_dict = {x: imgbatch, y1: y[0], y2: y[1], bbox: bbox_gr,
                             is_training: True}
                
                sess.run(train_step, feed_dict = feed_dict)
                if step % 100 == 0:
                    feed_dict[is_training] = False
                    print(sess.run(loss, feed_dict = feed_dict))
                step+= 1
            
            imgbatch = load_img(path, testLossList, imgSize)
            y, bbox_gr = cal_y(testLossList, imgInfo, anchorCoor)
            feed_dict = {x: imgbatch, y1: y[0], y2: y[1], bbox: bbox_gr,
                         is_training: False}
            testLoss.append(sess.run(loss, feed_dict = feed_dict))
            print('Test Loss:', testLoss[-1])
            timeE = time.time()
            print(epoch, timeE - timeS)
            if (fst_stage or warm_up) and epoch == epochMax-1:
                save = True
            if scd_stage and (epoch%1 == 0 or epoch == epochMax-1):
                AP, mAP = get_mAP(sess, x, decode_op, is_training, out, class_names, imgInfo, testList, path, imgSize, anchors, batchSize)
                if mAP > mAP0:
                    save = True
                    mAP0 = mAP
                    print('------------------save epoch: %d, mAP: %f-----------------------' %(epoch, mAP))
            if save:
                tf.train.Saver().save(sess, './Weights/%s/%s/%s' % (exp_name, save_name, save_name), epoch)
        np.save('./Weights/%s/%s/%s' % (exp_name, save_name, save_name), np.array(testLoss))
        sess.close()
        tf.reset_default_graph()

if __name__ == '__main__' :
    #div_train_test()
        
#    train_yolo(warm_up=True)
#    train_yolo(fst_stage=True)
#    train_yolo(scd_stage=True)
    train_yolo(predict=True)


