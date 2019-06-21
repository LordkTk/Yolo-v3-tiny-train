# Yolo-v3-tiny-train
train tiny-yolo-v3 for your own dataset

### Important reference: 
http://machinethink.net/blog/object-detection/; 

https://github.com/wizyoung/YOLOv3_TensorFlow; 

https://github.com/YunYang1994/tensorflow-yolov3/blob/master/core/yolov3.py

I learned a lot from the blog. It's fantastic and very suitable for beginners to know about yolo and its training. But during my own implementation i found there are two mistakes in the blog that will greatly influence yolo's training:
1. yolo uses warm-up during training, but the purpose is to let the model learn with a small rate in the begining, to avoid gradient explosion, not to 'encourage predictions to start matching the anchors for the detectors'. And if you do as the blog says, i.e., 'adding a fake ground-truth box in the center of each cell during early training steps', you will have great difficulty in reducing the loss.
2. yolo-v2 uses scale parameter to balance the influence of different losses, but yolo-v3 doesn't use such strategy. According to my own experience, training tiny-yolo-v3 needs focal loss. This can improve the model's performance.

### Function of the files:
1. 'dataset.py': preprocess the raw data. I train yolo for the voc dataset so the function is designed for this specifically. The output of the file is [[class, {xmin: , xmax: , ymin: , ymax: }], [W, H]].
2. 'k-means.py': with the processed dataset as input, calculate prior anchors using k-means.
3. 'prediction.py': convert weights file from darknet and store it as .ckpt file, for preparation for the subsequent training.
4. 'train_utils.py': various functions that will be used during the training process.
5. 'train.py': run this file to train yolo. There are lots of hyperparameters and may need to be modified according to your own dataset.

The layout of the files is shown below:

![Image text](imgs/file_layout.png)

### Parameters setting:

After experimenting with several groups of hyperparameters, i found that desirable results can be obtained with: 

--initial learning rate = 5e-4; 
 
--focal loss parameter: alpha = 0.25, gamma = 2; 
 
--epoch_num: warm-up: 2, first-stage: 4, second-stage: 20. 
 
The model gets an mAP of 32.8 for test-dataset (i get it by randomly dividing the full dataset to train and test parts, and the test part has 1000 imgs). i believe the present model can get much better results, but the process of adjusting parameters can be quite suffering. Besides, its performance can also be improved by using multiscale-training and data augmentation.

### Results:

Some of the results are shown below (with conf_threshold = 0.5 and iou_threshold = 0.3):

![Image text](imgs/0.jpg)

![Image text](imgs/1.jpg)

![Image text](imgs/2.jpg)

![Image text](imgs/3.jpg)

![Image text](imgs/4.jpg)

The detection results are much better than the those in my yolo-v3 repo (because of the different upsample methods between tf and darknet probably, see it in detail in my repo). 

Note it again that the model's performance can still be improved by adjusting parameters carefully, using multi-scale training and data augmentation.
