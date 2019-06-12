# Yolo-v3-tiny-train
train tiny-yolo-v3 for your own dataset

Important reference: https://link.zhihu.com/?target=http%3A//machinethink.net/blog/object-detection/

I learned a lot from this blog. It's fantastic and very suitable for beginners to know about yolo and its training. But during my own implementation i found there are two mistakes in the blog that will greatly influence yolo's training:
1. yolo uses warm-up during training, but the purpose is to let the model learn with a small rate in the begining, to avoid gradient explosion, not to 'encourage predictions to start matching the anchors for the detectors'. And if you do as the blog says, i.e., 'adding a fake ground-truth box in the center of each cell during early training steps', you will have great difficulty in reducing the loss.
2. yolo-v2 uses scale parameter to balance the influence of different losses, but yolo-v3 doesn't use such strategy. According to my own experience, training tiny-yolo-v3 needs focal loss. This can improve the model's performance a lot.

Function of the files:
1. 'dataset.py': preprocess the raw data. I train yolo for the voc dataset so the function is designed for this specifically. The output of the file is [[class, {xmin: , xmax: , ymin: , ymax: }], [W, H]].
2. 'k-means.py': with the processed dataset as input, calculate prior anchors using k-means.
3. 'prediction.py': convert weights file from darknet and store it as .ckpt file, for preparation for the subsequent training.
4. 'train_utils.py': various functions that will be used during the training process.
5. 'train.py': run this file to train yolo. There are lots of hyperparameters and may need to be modified according to your own dataset.

The layout of the files is shown below:
