#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 10:47:03
#   Description :
#
# ================================================================

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import core.utils as utils
import core.common as common
from core.config_tiny import cfg


class YOLOV3(object):
    """Implement tensoflow yolov3 here"""

    def __init__(self, input_data, trainable):

        self.trainable = trainable  # 训练
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)  # 类别
        self.num_class = len(self.classes)  # 类别长度
        self.strides = np.array(cfg.YOLO.STRIDES)  # 预测物体的大中小
        self.anchors = utils.get_anchors(cfg.YOLO.ANCHORS, is_tiny=True)  # 数据集的Anchors, tiny为2 x 3
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE  # Anchors 拉伸
        self.iou_loss_thresh = cfg.YOLO.IOU_LOSS_THRESH  # 阀值
        self.upsample_method = cfg.YOLO.UPSAMPLE_METHOD  # 上采样方法

        try:
            if cfg.YOLO.DSC:
                self.conv_lbbox, self.conv_mbbox = self.__build_nework_DSC(input_data)
            else:
                self.conv_lbbox, self.conv_mbbox = self.__build_nework(input_data)
        except:
            raise NotImplementedError("Can not build up YOLOv3-tiny network!")

        # 中物体
        with tf.variable_scope('pred_mbbox'):
            self.pred_mbbox = self.decode(self.conv_mbbox, self.anchors[0], self.strides[0])

        # 大物体
        with tf.variable_scope('pred_lbbox'):
            self.pred_lbbox = self.decode(self.conv_lbbox, self.anchors[1], self.strides[1])

    def separable_conv_block(self, input, dw_filter, output_channel, strides, name):
        """
        Params:
        input:
        filter:  a 4-D tuple: [filter_width, filter_height, in_channels, multiplier]
        output_channel: output channel of the separable_conv_block
        strides: a 4-D list: [1,strides,strides,1]
        """
        with tf.variable_scope(name):

            dw_weight = tf.get_variable(name='dw_filter', dtype=tf.float32, trainable=True,
                                     shape=dw_filter, initializer=tf.random_normal_initializer(stddev=0.01))

            dw = tf.nn.depthwise_conv2d(input=input, filter=dw_weight, strides=strides, padding="SAME", name='Conv/dw')

            bn_dw = tf.layers.batch_normalization(dw, beta_initializer=tf.zeros_initializer(),
                                                gamma_initializer=tf.ones_initializer(),
                                                moving_mean_initializer=tf.zeros_initializer(),
                                                moving_variance_initializer=tf.ones_initializer(), training=self.trainable,
                                                name='dw/bn')
            relu = tf.nn.leaky_relu(bn_dw,0.1)
            weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                     shape=(1, 1, dw_filter[2]*dw_filter[3], output_channel), initializer=tf.random_normal_initializer(stddev=0.01))

            conv = tf.nn.conv2d(input=relu, filter=weight, strides=[1, 1, 1, 1], padding="SAME",name="conv/s1")
            bn_pt = tf.layers.batch_normalization(conv, beta_initializer=tf.zeros_initializer(),
                                               gamma_initializer=tf.ones_initializer(),
                                               moving_mean_initializer=tf.zeros_initializer(),
                                               moving_variance_initializer=tf.ones_initializer(),
                                               training=self.trainable,
                                               name='pt/bn')
            return tf.nn.leaky_relu(bn_pt,0.1)

    def __build_nework_DSC(self, input_data):

        # mobilenet v1 backbone

        # input_data = tf.layers.conv2d(input_data,
        #                               filters=32,
        #                               kernel_size=(3, 3),
        #                               strides=(2,2),
        #                               padding = 'same',
        #                               activation = tf.nn.relu,
        #                               name = 'conv1'
        #                               )
        # input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(),
        #                                         gamma_initializer=tf.ones_initializer(),
        #                                         moving_mean_initializer=tf.zeros_initializer(),
        #                                         moving_variance_initializer=tf.ones_initializer(), training=self.trainable,
        #                                         name='bn')

        #input_data = common.convolutional(input_data, (3, 3, 3, 32), self.trainable, 'conv0')
        with tf.variable_scope('conv0'):
            weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                     shape=(3, 3, 3, 32), initializer=tf.random_normal_initializer(stddev=0.01))
            input_data = tf.nn.conv2d(input=input_data, filter=weight, strides=(1, 2, 2, 1), padding="SAME")


            input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(),
                                                 gamma_initializer=tf.ones_initializer(),
                                                 moving_mean_initializer=tf.zeros_initializer(),
                                                 moving_variance_initializer=tf.ones_initializer(),
                                                 training=self.trainable)


            input_data = tf.nn.leaky_relu(input_data, alpha=0.1)



        input_data = self.separable_conv_block(input=input_data, dw_filter=(3,3,32,1),output_channel=64,
                                                     strides=(1,1,1,1), name="spearable_1")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 64, 1), output_channel=128,
                                            strides=(1, 2, 2, 1), name="spearable_2")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3,3,128,1),output_channel=128,
                                                    strides=(1,1,1,1), name="spearable_3")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 128, 1), output_channel=256,
                                                    strides=(1, 2, 2, 1), name="spearable_4")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 256, 1), output_channel=256,
                                                strides=(1, 1, 1, 1), name="spearable_5")

        with tf.variable_scope("spearable_6"):
            dw_weight = tf.get_variable(name='dw_filter', dtype=tf.float32, trainable=True,
                                        shape=(3, 3, 256, 1), initializer=tf.random_normal_initializer(stddev=0.01))

            input_data = tf.nn.depthwise_conv2d(input=input_data, filter=dw_weight, strides=(1, 2, 2, 1), padding="SAME", name='Conv/dw')

            input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(),
                                                gamma_initializer=tf.ones_initializer(),
                                                moving_mean_initializer=tf.zeros_initializer(),
                                                moving_variance_initializer=tf.ones_initializer(), training=self.trainable,
                                                name='dw/bn')
            route = tf.nn.leaky_relu(input_data,0.1)

            weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                     shape=(1, 1, 256, 512), initializer=tf.random_normal_initializer(stddev=0.01))

            input_data = tf.nn.conv2d(input=route, filter=weight, strides=[1, 1, 1, 1], padding="SAME",name="conv/s1")
            input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(),
                                               gamma_initializer=tf.ones_initializer(),
                                               moving_mean_initializer=tf.zeros_initializer(),
                                               moving_variance_initializer=tf.ones_initializer(),
                                               training=self.trainable,
                                               name='pt/bn')
            input_data = tf.nn.leaky_relu(input_data,0.1)

        for i in range(5):
            input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 512, 1), output_channel=512,
                                                    strides=(1, 1, 1, 1), name="spearable_%d" % (i+ 7))
        
        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 512, 1), output_channel=1024,
                                        strides=(1, 2, 2, 1), name="spearable_12")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 1024, 1), output_channel=1024,
                                        strides=(1, 1, 1, 1), name="spearable_13")
        # mobilenet backbone end

        # yolo tiny backbone
        '''
        input_data = common.convolutional(input_data, (3, 3, 3, 16), self.trainable, 'conv0')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 16, 32), self.trainable, 'conv1')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 32, 64), self.trainable, 'conv2')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 64, 128), self.trainable, 'conv3')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv4')
        route = input_data
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv5')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv6')
        '''
        # end

        # yolo detect layer
        
        input_data = common.convolutional(input_data, (1, 1, 1024, 256), self.trainable, 'conv7')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                          self.trainable, 'conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv8')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route], axis=-1)

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 384, 256), self.trainable, 'conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                          self.trainable, 'conv_mbbox', activate=False, bn=False)
        
        # yolo detect layer end

        # yolo detect layer DSC
        '''
        input_data = self.separable_conv_block(input=input_data, dw_filter=(1, 1, 1024, 1),output_channel=256,
                                                     strides=(1,1,1,1), name="conv7")

        conv_lobj_branch = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 256, 1),output_channel=512,
                                                     strides=(1,1,1,1), name="conv_lobj_branch")
                                                     
        conv_lbbox = self.separable_conv_block(input=conv_lobj_branch, dw_filter=(1, 1, 512, 1),output_channel=3 * (self.num_class + 5),
                                                     strides=(1,1,1,1), name="conv_lbbox")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(1, 1, 256, 1),output_channel=128,
                                                     strides=(1,1,1,1), name="conv8")
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route], axis=-1)

        conv_mobj_branch = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 384, 1),output_channel=256,
                                                     strides=(1,1,1,1), name="conv_mobj_branch")
        conv_mbbox = self.separable_conv_block(input=conv_mobj_branch, dw_filter=(1, 1, 256, 1),output_channel=3 * (self.num_class + 5),
                                                     strides=(1,1,1,1), name="conv_mbbox")
        '''

        return conv_lbbox, conv_mbbox

    def __build_nework(self, input_data):

        # mobilenet v1 backbone

        # input_data = tf.layers.conv2d(input_data,
        #                               filters=32,
        #                               kernel_size=(3, 3),
        #                               strides=(2,2),
        #                               padding = 'same',
        #                               activation = tf.nn.relu,
        #                               name = 'conv1'
        #                               )
        # input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(),
        #                                         gamma_initializer=tf.ones_initializer(),
        #                                         moving_mean_initializer=tf.zeros_initializer(),
        #                                         moving_variance_initializer=tf.ones_initializer(), training=self.trainable,
        #                                         name='bn')
        '''
        input_data = common.convolutional(input_data, (3, 3, 3, 32), self.trainable, 'conv0')

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3,3,32,1),output_channel=64,
                                                     strides=(1,1,1,1), name="spearable_1")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 64, 1), output_channel=128,
                                            strides=(1, 2, 2, 1), name="spearable_2")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3,3,128,1),output_channel=128,
                                                    strides=(1,1,1,1), name="spearable_3")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 128, 1), output_channel=256,
                                                    strides=(1, 2, 2, 1), name="spearable_4")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 256, 1), output_channel=256,
                                                strides=(1, 1, 1, 1), name="spearable_5")

        with tf.variable_scope("spearable_6"):
            dw_weight = tf.get_variable(name='dw_filter', dtype=tf.float32, trainable=True,
                                        shape=(3, 3, 256, 1), initializer=tf.random_normal_initializer(stddev=0.01))

            input_data = tf.nn.depthwise_conv2d(input=input_data, filter=dw_weight, strides=(1, 2, 2, 1), padding="SAME", name='Conv/dw')

            input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(),
                                                gamma_initializer=tf.ones_initializer(),
                                                moving_mean_initializer=tf.zeros_initializer(),
                                                moving_variance_initializer=tf.ones_initializer(), training=self.trainable,
                                                name='dw/bn')
            route = tf.nn.leaky_relu(input_data,0.1)

            weight = tf.get_variable(name='weight', dtype=tf.float32, trainable=True,
                                     shape=(1, 1, 256, 512), initializer=tf.random_normal_initializer(stddev=0.01))

            input_data = tf.nn.conv2d(input=route, filter=weight, strides=[1, 1, 1, 1], padding="SAME",name="conv/s1")
            input_data = tf.layers.batch_normalization(input_data, beta_initializer=tf.zeros_initializer(),
                                               gamma_initializer=tf.ones_initializer(),
                                               moving_mean_initializer=tf.zeros_initializer(),
                                               moving_variance_initializer=tf.ones_initializer(),
                                               training=self.trainable,
                                               name='pt/bn')
            input_data = tf.nn.leaky_relu(input_data,0.1)

        for i in range(5):
            input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 512, 1), output_channel=512,
                                                    strides=(1, 1, 1, 1), name="spearable_%d" % (i+ 7))
        
        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 512, 1), output_channel=1024,
                                        strides=(1, 2, 2, 1), name="spearable_12")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 1024, 1), output_channel=1024,
                                        strides=(1, 1, 1, 1), name="spearable_13")
        '''
        # mobilenet backbone end

        # yolo tiny backbone
        
        input_data = common.convolutional(input_data, (3, 3, 3, 16), self.trainable, 'conv0')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 16, 32), self.trainable, 'conv1')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 32, 64), self.trainable, 'conv2')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 64, 128), self.trainable, 'conv3')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 128, 256), self.trainable, 'conv4')
        route = input_data
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv5')
        input_data = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        input_data = common.convolutional(input_data, (3, 3, 512, 1024), self.trainable, 'conv6')
        
        # end

        # yolo detect layer
        
        input_data = common.convolutional(input_data, (1, 1, 1024, 256), self.trainable, 'conv7')

        conv_lobj_branch = common.convolutional(input_data, (3, 3, 256, 512), self.trainable, 'conv_lobj_branch')
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (self.num_class + 5)),
                                          self.trainable, 'conv_lbbox', activate=False, bn=False)

        input_data = common.convolutional(input_data, (1, 1, 256, 128), self.trainable, 'conv8')
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route], axis=-1)

        conv_mobj_branch = common.convolutional(input_data, (3, 3, 384, 256), self.trainable, 'conv_mobj_branch')
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (self.num_class + 5)),
                                          self.trainable, 'conv_mbbox', activate=False, bn=False)
        
        # yolo detect layer end

        # yolo detect layer DSC
        '''
        input_data = self.separable_conv_block(input=input_data, dw_filter=(1, 1, 1024, 1),output_channel=256,
                                                     strides=(1,1,1,1), name="conv7")

        conv_lobj_branch = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 256, 1),output_channel=512,
                                                     strides=(1,1,1,1), name="conv_lobj_branch")
                                                     
        conv_lbbox = self.separable_conv_block(input=conv_lobj_branch, dw_filter=(1, 1, 512, 1),output_channel=3 * (self.num_class + 5),
                                                     strides=(1,1,1,1), name="conv_lbbox")

        input_data = self.separable_conv_block(input=input_data, dw_filter=(1, 1, 256, 1),output_channel=128,
                                                     strides=(1,1,1,1), name="conv8")
        input_data = common.upsample(input_data, name='upsample0', method=self.upsample_method)

        with tf.variable_scope('route_1'):
            input_data = tf.concat([input_data, route], axis=-1)

        conv_mobj_branch = self.separable_conv_block(input=input_data, dw_filter=(3, 3, 384, 1),output_channel=256,
                                                     strides=(1,1,1,1), name="conv_mobj_branch")
        conv_mbbox = self.separable_conv_block(input=conv_mobj_branch, dw_filter=(1, 1, 256, 1),output_channel=3 * (self.num_class + 5),
                                                     strides=(1,1,1,1), name="conv_mbbox")
        '''

        return conv_lbbox, conv_mbbox

    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """

        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output,
                                 (batch_size, output_size, output_size, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(output_size, dtype=tf.int32)[:, tf.newaxis], [1, output_size])
        x = tf.tile(tf.range(output_size, dtype=tf.int32)[tf.newaxis, :], [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def focal(self, target, actual, alpha=1, gamma=2):
        focal_loss = alpha * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def bbox_giou(self, boxes1, boxes2):

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

    def bbox_iou(self, boxes1, boxes2):

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        iou = 1.0 * inter_area / union_area

        return iou

    def loss_layer(self, conv, pred, label, bboxes, anchors, stride):

        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = stride * output_size
        conv = tf.reshape(conv, (batch_size, output_size, output_size,
                                 self.anchor_per_scale, 5 + self.num_class))
        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.iou_loss_thresh, tf.float32)

        conf_focal = self.focal(respond_bbox, pred_conf)

        conf_loss = conf_focal * (
                respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
                +
                respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
        )

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self, label_mbbox, label_lbbox, true_mbbox, true_lbbox):

        with tf.name_scope('medium_box_loss'):
            loss_mbbox = self.loss_layer(self.conv_mbbox, self.pred_mbbox, label_mbbox, true_mbbox,
                                         anchors=self.anchors[0], stride=self.strides[0])

        with tf.name_scope('bigger_box_loss'):
            loss_lbbox = self.loss_layer(self.conv_lbbox, self.pred_lbbox, label_lbbox, true_lbbox,
                                         anchors=self.anchors[1], stride=self.strides[1])

        with tf.name_scope('giou_loss'):
            giou_loss = loss_mbbox[0] + loss_lbbox[0]

        with tf.name_scope('conf_loss'):
            conf_loss = loss_mbbox[1] + loss_lbbox[1]

        with tf.name_scope('prob_loss'):
            prob_loss = loss_mbbox[2] + loss_lbbox[2]

        return giou_loss, conf_loss, prob_loss
