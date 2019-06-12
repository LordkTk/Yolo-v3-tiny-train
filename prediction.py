'comment: for v3, useful data start from 6th element'

import numpy as np
import tensorflow as tf
import cv2

#anchors = np.array([35.3655,52.1796, 89.6969,130.258, 111.745,236.696, 295.59,190.8, 185.47,341.823, 358.706,361.342]).reshape([6, 2])

def decode(out, imgSize, num_classes=80, anchorTotal=None):
    bboxesTotal = []; obj_probsTotal = []; class_probsTotal = []
    for i, detection_feat in enumerate(out):
        _, H, W, _ = detection_feat.get_shape().as_list()
        anchors = anchorTotal[(1-i)*3:(1-i)*3+3, :]
        num_anchors = len(anchors)
        
        detetion_results = tf.reshape(detection_feat, [-1, H * W, num_anchors, num_classes + 5])
        
        bbox_xy = tf.nn.sigmoid(detetion_results[:, :, :, :2])
        bbox_wh = tf.exp(detetion_results[:, :, :, 2:4])
        obj_probs = tf.nn.sigmoid(detetion_results[:, :, :, 4])
        class_probs = tf.nn.sigmoid(detetion_results[:, :, :, 5:])
    
        anchors = tf.constant(anchors, dtype=tf.float32)
    
        height_ind = tf.range(H, dtype=tf.float32)
        width_ind = tf.range(W, dtype=tf.float32)
        
        x_offset, y_offset = tf.meshgrid(height_ind, width_ind)
        x_offset = tf.reshape(x_offset, [1, -1, 1])
        y_offset = tf.reshape(y_offset, [1, -1, 1])
        
        bbox_x = (bbox_xy[:, :, :, 0] + x_offset) / W
        bbox_y = (bbox_xy[:, :, :, 1] + y_offset) / H
        bbox_w = bbox_wh[:, :, :, 0] * anchors[:, 0] / imgSize * 0.5
        bbox_h = bbox_wh[:, :, :, 1] * anchors[:, 1] / imgSize * 0.5

        bboxes = tf.stack([bbox_x - bbox_w, bbox_y - bbox_h, bbox_x + bbox_w, bbox_y + bbox_h], axis=3)
        bboxesTotal.append(bboxes)
        obj_probsTotal.append(obj_probs)
        class_probsTotal.append(class_probs)
    return tf.concat(bboxesTotal, axis=1), tf.concat(obj_probsTotal, axis=1), tf.concat(class_probsTotal, axis=1)

def conv2d(x, outfilters, name, ind, tiny, is_training, size=3, stride=1, batchnorm=True, trainable=False):
    _,_,_, infilters = x.get_shape().as_list()
    if batchnorm == True:
        beta, gamma, mean, var = tiny[ind:ind+4*outfilters].reshape([4, outfilters])##
        ind = ind+4*outfilters
    else:
        b = tiny[ind:ind+outfilters]
        bias = tf.Variable(b, name = 'b'+name, trainable = trainable)
        ind = ind + outfilters
    num = size*size*infilters*outfilters
    w = np.transpose(tiny[ind:ind+num].reshape([outfilters, infilters, size, size]), (2,3,1,0))
    Weights = tf.Variable(w, name = 'W'+name, trainable = trainable)
    ind = ind + num
    if batchnorm == True:
        xx = tf.nn.conv2d(x, Weights, [1,stride,stride,1], 'SAME')
        xx = tf.contrib.layers.batch_norm(xx, scale=True, 
                                          param_initializers={'beta':tf.constant_initializer(beta), 
                                                              'gamma':tf.constant_initializer(gamma), 
                                                              'moving_mean':tf.constant_initializer(mean), 
                                                              'moving_variance':tf.constant_initializer(var)}, 
                                                              is_training = is_training, 
                                                              trainable = trainable)
        return tf.nn.leaky_relu(xx, 0.1), ind
    else:
        return tf.nn.conv2d(x, Weights, [1, stride,stride,1], 'SAME') + bias, ind
def max_pool(x, size=2, stride=2):
    if stride == 2:
        return tf.nn.max_pool(x, [1,size,size,1], [1,stride,stride,1], 'VALID')
    else:
        return tf.nn.max_pool(x, [1,size,size,1], [1,stride,stride,1], 'SAME')
def route(x1, x2):
    [_, H, W, _]= x1.get_shape().as_list()
    x1 = tf.image.resize_nearest_neighbor(x1, [H*2, W*2])
    return tf.concat([x1, x2], axis=3)

def build_darknet(x, tiny, is_training):
    ind = 0
    
    net, ind = conv2d(x, 16, '_conv1', ind, tiny, is_training)
    net = max_pool(net)
    net, ind = conv2d(net, 32, '_conv2', ind, tiny, is_training)
    net = max_pool(net)
    net, ind = conv2d(net, 64, '_conv3', ind, tiny, is_training)
    net = max_pool(net)
    net, ind = conv2d(net, 128, '_conv4', ind, tiny, is_training)
    net = max_pool(net)
    net, ind = conv2d(net, 256, '_conv5', ind, tiny, is_training)
    route2 = net
    net = max_pool(net)
    net, ind = conv2d(net, 512, '_conv6', ind, tiny, is_training)
    net = max_pool(net, stride=1)
    net, ind = conv2d(net, 1024, '_conv7', ind, tiny, is_training)
    return net, ind, route2
def build_detnet(net, ind, route2, tiny, is_training, trainable=True):
    out = []
    #det1
    net, ind = conv2d(net, 256, '_det1_1', ind, tiny, is_training, size=1, trainable=True)
    route1 = net
    net, ind = conv2d(net, 512, '_det1_2', ind, tiny, is_training, trainable=True)
    out1, ind = conv2d(net, 255, '_det1_3', ind, tiny, is_training, size=1, batchnorm=False, trainable=True)
    out.append(out1)
    
    #det2
    net, ind = conv2d(route1, 128, '_det2_1', ind, tiny, is_training, size=1, trainable=True)
    net = route(net, route2)
    net, ind = conv2d(net, 256, '_det2_2', ind, tiny, is_training, trainable=True)
    out2, ind = conv2d(net, 255, '_det2_3', ind, tiny, is_training, size=1, batchnorm=False, trainable=True)
    out.append(out2)
    return out, ind
def post_process(sess, x, is_training, out, pathList, imgSize, num_classes, class_names, anchors, save_weights=False):
    def arg_max(x):
        xmax = 0
        id1 = 0
        id2 = 0
        w1, w2 = x.shape
        for i in range(w1):
            for j in range(w2):
                if x[i, j]>xmax:
                    xmax = x[i,j]
                    id1 = i
                    id2 = j
        return id1, id2
    imgTotal = []
    imgSrcTotal = []
    for path in pathList:
        imgSrc = cv2.imread(path, 1)
        img = np.float32(cv2.cvtColor(imgSrc, cv2.COLOR_BGR2RGB)/255)
        img = cv2.resize(img, (imgSize, imgSize))[np.newaxis, :,:,:]
        imgTotal.append(img)
        imgSrcTotal.append(imgSrc)
    imgTotal = np.concatenate(imgTotal, axis=0)
        
    bboxTotal, obj_probsTotal, class_probsTotal = sess.run(decode(out, imgSize, num_classes, anchorTotal=anchors), feed_dict={x:imgTotal, is_training:False})
    if save_weights:
        saver = tf.train.Saver()
        saver.save(sess, './Weights/weightsInit.ckpt')
    
    for i, imgSrc in enumerate(imgSrcTotal):
        H, W, _ = imgSrc.shape
        #[:, 3]; [:, 3, 4]
        class_name = np.argmax(class_probsTotal[i], axis = 2)
        class_probs = np.max(class_probsTotal[i], axis = 2)
        obj_probs = obj_probsTotal[i]
        bbox = bboxTotal[i]
        bbox[bbox>1] = 1
        bbox[bbox<0] = 0
        confidence = class_probs * obj_probs
        conf = confidence.copy()
        confidence[confidence<0.5] = 0 ######################################################################################
        
        det_indTotal = []
        while (np.max(confidence)!=0):
            id1, id2 = arg_max(confidence)
            bbox1_area = (bbox[id1, id2,3]-bbox[id1, id2,1])*(bbox[id1, id2,2]-bbox[id1, id2,0])
            sign = 0
            for coor in det_indTotal:
                if coor == None:
                    det_indTotal.append([id1, id2])
                else:
                    xi1 = max(bbox[id1, id2, 0], bbox[coor[0], coor[1], 0])
                    yi1 = max(bbox[id1, id2, 1], bbox[coor[0], coor[1], 1])
                    xi2 = min(bbox[id1, id2, 2], bbox[coor[0], coor[1], 2])
                    yi2 = min(bbox[id1, id2, 3], bbox[coor[0], coor[1], 3])
                    int_area = (max(yi2, 0) - max(yi1, 0)) * (max(xi2, 0) - max(xi1, 0))
                    bbox2_area = (bbox[coor[0],coor[1],3]-bbox[coor[0],coor[1],1]) * (bbox[coor[0],coor[1],2]-bbox[coor[0],coor[1],0])
                    uni_area = bbox1_area + bbox2_area - int_area
                    iou = int_area/uni_area
                    if iou>0.3: ###########################################################################################
                        sign = 1
                        break
            if sign==0:
                det_indTotal.append([id1, id2]) 
            confidence[id1, id2] = 0
            
        depict = []
        for [id1, id2] in det_indTotal:
            x1,y1,x2,y2 = bbox[id1, id2]
            x1 = int(x1*W); x2 = int(x2*W); y1 = int(y1*H); y2 = int(y2*H)
            name = class_names[class_name[id1, id2]]
            depict.append([x1,y1,x2,y2])
            cv2.rectangle(imgSrc, (x1,y1), (x2,y2), (255,0,0), 2)
            
            y = y1-8 if y1-8>8 else y1+8
            cv2.putText(imgSrc, '%s %.2f' %(name, conf[id1, id2]), (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
        cv2.imshow('%d' %i, imgSrc)
def get_class_names(path):
    info = open(path)
    filedata = info.readlines()
    class_names = []
    for line in filedata:
        line.strip()
        class_names.append(line)
    return class_names
if __name__ == '__main__' :
    imgSize = 416
    tiny = np.fromfile('yolov3-tiny.weights', np.float32)[5:]
    anchors = np.array([10,14,  23,27,  37,58,  81,82,  135,169,  344,319]).reshape([6,2])
    class_names = get_class_names('./coconames.txt')
    
    x = tf.placeholder(tf.float32, [None,imgSize,imgSize,3])
    is_training = tf.placeholder(tf.bool)
    
    net, ind, route2 = build_darknet(x, tiny, is_training)
    out, ind = build_detnet(net, ind, route2, tiny, is_training)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    pathList = ['C:/Users/cfd_Liu/Desktop/Machine Learning/Code/PracticeCode/TensorFlow Learning/OpenCV/LaneDet/YOLO/img/sample_car.jpg']
    post_process(sess, x, is_training, out, pathList, imgSize, 80, class_names, anchors, save_weights=False)
    
