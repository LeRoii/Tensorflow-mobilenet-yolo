#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
# ================================================================

import os
import time
import shutil
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import core.utils as utils
from tqdm import tqdm
from core.dataset_tiny import Dataset
from core.yolov3_tiny import YOLOV3
from core.config_tiny import cfg


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE  # Anchors 拉伸
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)  # 类别名称
        self.num_classes = len(self.classes)  # 类别长度
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT  # 初始学习率
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END  # 终止学习率
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS  # 第一部分批次大小
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS  # 第二部分批次大小
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS  # warmup 批次
        self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT  # 预训练权重
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))  # 保存批次的时间
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY  # decay, 防止过拟合
        self.max_bbox_per_scale = 150
        self.train_log_dir = "./data/log/train"  # log 保存目录
        self.train_set = Dataset('train')
        self.test_set = Dataset('test')
        self.steps_per_period = len(self.train_set)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))  # 启动会话

        with tf.name_scope('define_input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                self.label_mbbox, self.label_lbbox,
                self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                       dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                      dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                 (1 + tf.cos(
                                     (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        # with tf.name_scope('loader_and_saver'):
        #     self.loader = tf.train.Saver(self.net_var)
        #     self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                     var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        # with tf.name_scope('loader_and_saver'):
        #     self.loader = tf.train.Saver(self.net_var)
        #     self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss", self.giou_loss)
            tf.summary.scalar("conf_loss", self.conf_loss)
            tf.summary.scalar("prob_loss", self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            lod_dir = "./data/log/"
            if os.path.exists(lod_dir):
                shutil.rmtree(lod_dir)
            os.mkdir(lod_dir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(lod_dir, graph=self.sess.graph)

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.train_set)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                        self.input_data: train_data[0],
                        self.label_mbbox: train_data[1],
                        self.label_lbbox: train_data[2],
                        self.true_mbboxes: train_data[3],
                        self.true_lbboxes: train_data[4],
                        self.trainable: True,
                    })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.6f， lr:%.5f" % (train_step_loss, self.learn_rate))

            for test_data in self.test_set:
                test_step_loss = self.sess.run(self.loss, feed_dict={
                    self.input_data: test_data[0],
                    self.label_mbbox: test_data[1],
                    self.label_lbbox: test_data[2],
                    self.true_mbboxes: test_data[3],
                    self.true_lbboxes: test_data[4],
                    self.trainable: False,
                })

                test_epoch_loss.append(test_step_loss)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/yolov3_mn416_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=epoch)


if __name__ == '__main__':
    YoloTrain().train()
