# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:01:44 2019

@author: cfd_Liu
"""

import tensorflow as tf
import numpy as np
import random
import os

def div_train_test():
    path = os.path.abspath(r'C:\Users\cfd_Liu\Desktop\Machine Learning\Code\PracticeCode\TensorFlow Learning\OpenCV\LaneDet\YOLO\yolo_train\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages')
    dirlist = os.listdir(path)
    testList = []
    trainList = dirlist.copy()
    randint = random.sample(range(0, len(dirlist)), 1000)
    for i in randint:
        name = dirlist[i]
        testList.append(name)
    for i in range(len(randint)):
        trainList.remove(testList[i])
    np.save('./data/trainList', {1:trainList})
    np.save('./data/testList', {1:testList})
    np.save('./data/path', {1:path})
    
def conv2d(x, outfilters, name, is_training, size=3, stride=1, batchnorm=True, trainable=False):
    _,_,_, infilters = x.get_shape().as_list()
    if batchnorm == False:
        bias = tf.Variable(tf.zeros([outfilters]) + 0.1, 
                           name = 'b'+name, trainable = trainable)
    ele_num = infilters * outfilters * size**2
    Weights = tf.Variable(tf.random_normal([size, size, infilters, outfilters]) * (2 / ele_num)**0.5, 
                          name = 'W'+name, trainable = trainable)
    if batchnorm == True:
        xx = tf.nn.conv2d(x, Weights, [1,stride,stride,1], 'SAME')
        xx = tf.contrib.layers.batch_norm(xx, scale=True, 
                                          is_training = is_training, trainable = trainable)
        return tf.nn.leaky_relu(xx, 0.1)
    else:
        return tf.nn.conv2d(x, Weights, [1, stride,stride,1], 'SAME') + bias
def max_pool(x, size=2, stride=2):
    if stride == 2:
        return tf.nn.max_pool(x, [1,size,size,1], [1,stride,stride,1], 'VALID')
    else:
        return tf.nn.max_pool(x, [1,size,size,1], [1,stride,stride,1], 'SAME')
def route(x1, x2):
    [_, H, W, _]= x1.get_shape().as_list()
    x1 = tf.image.resize_nearest_neighbor(x1, [H*2, W*2])
    return tf.concat([x1, x2], axis=3)

def build_darknet(x, is_training, trainable=False):
    
    net = conv2d(x, 16, '_conv1', is_training, trainable=trainable)
    net = max_pool(net)
    net = conv2d(net, 32, '_conv2', is_training, trainable=trainable)
    net = max_pool(net)
    net = conv2d(net, 64, '_conv3', is_training, trainable=trainable)
    net = max_pool(net)
    net = conv2d(net, 128, '_conv4', is_training, trainable=trainable)
    net = max_pool(net)
    net = conv2d(net, 256, '_conv5', is_training, trainable=trainable)
    route2 = net
    net = max_pool(net)
    net = conv2d(net, 512, '_conv6', is_training, trainable=trainable)
    net = max_pool(net, stride=1)
    net = conv2d(net, 1024, '_conv7', is_training, trainable=trainable)
    return net, route2

def build_detnet(net, route2, is_training, trainable=True):
    out = []
    #det1
    net = conv2d(net, 256, '_det1_1', is_training, size=1, trainable=trainable)
    route1 = net
    net = conv2d(net, 512, '_det1_2', is_training, trainable=trainable)
    out1 = conv2d(net, 75, '_det1_3', is_training, size=1, batchnorm=False, trainable=trainable)
    out.append(out1)
    
    #det2
    net = conv2d(route1, 128, '_det2_1', is_training, size=1, trainable=trainable)
    net = route(net, route2)
    net = conv2d(net, 256, '_det2_2', is_training, trainable=trainable)
    out2 = conv2d(net, 75, '_det2_3', is_training, size=1, batchnorm=False, trainable=trainable)
    out.append(out2)
    return out
def proc_anchors(anchors, imgSize):
    anchorCoor = [np.zeros([13, 13, 3, 4], np.float32), np.zeros([26, 26, 3, 4], np.float32)]
    anchor = [anchors[3:6, :] / imgSize, anchors[:3, :] / imgSize]
    for n in range(2):
        cellNum = (n+1) * 13
        for i in range(cellNum):
            for j in range(cellNum):
                for k in range(3):
                    y_cen = 1/cellNum * i + 1/cellNum/2
                    x_cen = 1/cellNum * j + 1/cellNum/2
                    xmin = x_cen - anchor[n][k, 0]/2
                    ymin = y_cen - anchor[n][k, 1]/2
                    xmax = x_cen + anchor[n][k, 0]/2
                    ymax = y_cen + anchor[n][k, 1]/2
                    anchorCoor[n][i, j, k] = [xmin, ymin, xmax, ymax]#after normalization
    return anchorCoor
def cal_iou(bbox1, bbox2):
    xi1 = max(bbox1[0], bbox2[0])
    yi1 = max(bbox1[1], bbox2[1])
    xi2 = min(bbox1[2], bbox2[2])
    yi2 = min(bbox1[3], bbox2[3])
    int_area = (max(yi2, 0) - max(yi1, 0)) * (max(xi2, 0) - max(xi1, 0))
    bbox1_area = (bbox1[3]-bbox1[1])*(bbox1[2]-bbox1[0])
    bbox2_area = (bbox2[3]-bbox2[1])*(bbox2[2]-bbox2[0])
    uni_area = bbox1_area + bbox2_area - int_area
    iou = int_area/ uni_area
    return iou
def cal_wh(bbox_gr, bbox_pri):
    w = (bbox_gr[2] - bbox_gr[0])/(bbox_pri[2] - bbox_pri[0])
    h = (bbox_gr[3] - bbox_gr[1])/(bbox_pri[3] - bbox_pri[1])
    return np.log(w), np.log(h)
def proc_bbox_batch(bbox_batch):
    maxnum = 0
    for bbox_img in bbox_batch:
        maxnum = len(bbox_img) if len(bbox_img)>maxnum else maxnum
    bbox = np.zeros([len(bbox_batch), maxnum, 4], np.float32)
    for i, bbox_img in enumerate(bbox_batch):
        for j, each_bbox in enumerate(bbox_img):
            bbox[i,j] = np.array(each_bbox, np.float32)
    return bbox #[x_cen, y_cen, w, h] in range (0,1)
def cal_y(batchList, imgInfo, anchorCoor):
    batchSize = len(batchList)
    y = [np.zeros([batchSize, 13, 13, 3, 25], np.float32), np.zeros([batchSize, 26, 26, 3, 25], np.float32)]
    bbox_batch = []
    for img_ind, img in enumerate(batchList):
        Info = imgInfo[img].copy()
        w, h = Info[-1]
        Info.remove([w, h])
        bbox_img = []
        for objInfo in Info:
            iou_log = np.zeros([2, 3, 9], np.float32)
            class_ind = objInfo[0]
            bbox_gr = [objInfo[1]['xmin']/w, objInfo[1]['ymin']/h, objInfo[1]['xmax']/w, objInfo[1]['ymax']/h] # in range (0, 1)
            bbox_img.append([(bbox_gr[2]+bbox_gr[0])/2, (bbox_gr[3]+bbox_gr[1])/2, (bbox_gr[2]-bbox_gr[0]), (bbox_gr[3]-bbox_gr[1])])
            x_cen_gr = (bbox_gr[0] + bbox_gr[2]) / 2.
            y_cen_gr = (bbox_gr[1] + bbox_gr[3]) / 2.
            for n in range(2):
                cellNum = 13 * (n + 1)
                x_cell_ind = int(x_cen_gr * cellNum // 1)
                y_cell_ind = int(y_cen_gr * cellNum // 1)
                x_cen = x_cen_gr * cellNum - x_cell_ind
                y_cen = y_cen_gr * cellNum - y_cell_ind
                for anch in range(3):
                    bbox_pri = anchorCoor[n][y_cell_ind, x_cell_ind, anch]
                    iou = cal_iou(bbox_pri, bbox_gr)
                    w_gr, h_gr = cal_wh(bbox_gr, bbox_pri)
                    iou_log[n, anch] = [iou, x_cell_ind, y_cell_ind, x_cen, y_cen, w_gr, h_gr, x_cen_gr, y_cen_gr]
                    
            p1 = 0; p2 = 0; t = iou_log[0,0,0]
            for i in range(2):
                for j in range(3):
                    if iou_log[i, j, 0] > t:
                        p1 = i; p2 = j; t = iou_log[i, j, 0]
                        
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, class_ind + 5] = 1
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 4] = 1
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 0] = iou_log[p1, p2, 3]
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 1] = iou_log[p1, p2, 4]
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 2] = iou_log[p1, p2, 5]
            y[p1][img_ind, int(iou_log[p1, p2, 2]), int(iou_log[p1, p2, 1]), p2, 3] = iou_log[p1, p2, 6]
        bbox_batch.append(bbox_img)
    return y, proc_bbox_batch(bbox_batch)
    
def cal_loss(out_pre, out_gr, anchorCoor, bbox, no_obj_scale, coor_scale, obj_scale, class_scale, focal_loss=False):
    #[2][none, 13, 13, 75]
    #[2][None, 13, 13, 3, 25]    
        
    loss = tf.constant(0, tf.float32)
    for n in range(2):
        cellNum = 13 * (n+1)
        y_pre = tf.reshape(out_pre[n], [-1, cellNum, cellNum, 3, 25])
        y_gr = out_gr[n]
        #[N, 13, 13, 3]
        mask = y_gr[:,:,:,:, 4]
        
        mask_coor = tf.stack([mask, mask], axis = 4)
        coor_xy_pre = tf.sigmoid(y_pre[:,:,:,:, :2])
        coor_xy_gr = y_gr[:,:,:,:, :2]
        coor_wh_pre = y_pre[:,:,:,:, 2:4]
        coor_wh_gr = y_gr[:,:,:,:, 2:4]
        coor_scale_mod = (2 - tf.exp(coor_wh_gr[..., 0:1]) * 
                          (anchorCoor[n][..., 2:3]-anchorCoor[n][..., :1]) * tf.exp(coor_wh_gr[..., 1:2]) * 
                          (anchorCoor[n][..., 3:4] - anchorCoor[n][..., 1:2])) * coor_scale
        coor_xy_loss = tf.reduce_sum(tf.reduce_mean(tf.square(coor_xy_pre - coor_xy_gr) * mask_coor * coor_scale_mod, axis=0))
        coor_wh_loss = tf.reduce_sum(tf.reduce_mean(tf.square(coor_wh_pre - coor_wh_gr) * mask_coor * coor_scale_mod, axis=0))
        
        ignore_mask = cal_ignore_mask(y_pre[..., :4], bbox, anchorCoor[n], cellNum)
        mask_no_obj = (1 - mask) * ignore_mask
        obj_pre = y_pre[:,:,:,:, 4]
        obj_gr = y_gr[:,:,:,:, 4]

        if focal_loss:
            fl_obj = 0.25 * tf.pow((1 - tf.sigmoid(obj_pre)), 2)
            fl_no_obj = 0.75 * tf.pow(tf.sigmoid(obj_pre), 2)
        else:
            fl_obj = 1
            fl_no_obj = 1
        no_obj_loss = tf.reduce_sum(
                tf.reduce_mean(
                        fl_no_obj * tf.nn.sigmoid_cross_entropy_with_logits(labels = obj_gr, logits = obj_pre) * mask_no_obj * no_obj_scale, axis=0))
        
        obj_loss = tf.reduce_sum(
                tf.reduce_mean(
                        fl_obj * tf.nn.sigmoid_cross_entropy_with_logits(labels = obj_gr, logits = obj_pre) * mask * obj_scale, axis=0))
        
        mask_class = tf.stack([mask for i in range(20)], axis = 4)
        class_pre = y_pre[:,:,:,:, 5:]
        class_gr = y_gr[:,:,:,:, 5:]
        class_loss = tf.reduce_sum(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = class_gr, logits = class_pre) * mask_class * class_scale, axis=0))
        
        loss = loss + coor_xy_loss + coor_wh_loss + no_obj_loss + obj_loss + class_loss
    return [loss, coor_xy_loss, coor_wh_loss, no_obj_loss, obj_loss, class_loss]

def cal_ignore_mask(bbox_pre, bbox_gr, anchor, cellNum):
    height_ind = tf.range(cellNum, dtype=tf.float32)
    x_offset, y_offset = tf.meshgrid(height_ind, height_ind)
    #[N,13,13,3]
    bbox_pre_x = (tf.sigmoid(bbox_pre[..., 0]) + x_offset[:,:,np.newaxis]) / cellNum
    bbox_pre_y = (tf.sigmoid(bbox_pre[..., 1]) + y_offset[:,:,np.newaxis]) / cellNum
    bbox_pre_w = tf.exp(bbox_pre[..., 2]) * (anchor[..., 2] - anchor[..., 0])
    bbox_pre_h = tf.exp(bbox_pre[..., 3]) * (anchor[..., 3] - anchor[..., 1])
    #[N,13,13,3,4]
    bbox_pre = tf.stack([bbox_pre_x, bbox_pre_y, bbox_pre_w, bbox_pre_h], axis=4)
    #[N,13,13,3,M,4]
    bbox_pre = bbox_pre[:,:,:,:, np.newaxis, :]
    bbox_gr = bbox_gr[:, np.newaxis, np.newaxis, np.newaxis, :, :]
    
    bbox_pre_area = bbox_pre[..., 2] * bbox_pre[..., 3]
    bbox_gr_area = bbox_gr[..., 2] * bbox_gr[..., 3]

    bbox_pre = tf.concat([bbox_pre[..., :2] - bbox_pre[..., 2:] * 0.5,
                        bbox_pre[..., :2] + bbox_pre[..., 2:] * 0.5], axis=-1)
    bbox_gr = tf.concat([bbox_gr[..., :2] - bbox_gr[..., 2:] * 0.5,
                        bbox_gr[..., :2] + bbox_gr[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(bbox_pre[..., :2], bbox_gr[..., :2])
    right_down = tf.minimum(bbox_pre[..., 2:], bbox_gr[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    #[N, 13, 13, 3, M]
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = bbox_pre_area + bbox_gr_area - inter_area
    iou = 1.0 * inter_area / union_area
    #[N, 13, 13, 3]
    max_iou = tf.reduce_max(iou, axis=-1)
    return tf.cast(max_iou < 0.5, tf.float32)

def bbox_giou(boxes1, boxes2):

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
    enclose_area = enclose[..., 0] * enclose[..., 1]
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    return giou