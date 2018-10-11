# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import _init_paths

import cv2
import time
import argparse
import logging
import pprint
import os
import sys
from config.config import config, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='Train R-FCN network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', help='frequency of logging', default=config.default.frequent, type=int)
    args = parser.parse_args()
    return args


args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import shutil
import numpy as np
import mxnet as mx

from symbols import *
from core import callback, metric
# from core.loader import AnchorLoader
from core.loader_ucf101 import TrainLoader
from core.module import MutableModule
from core.moduleway import WayModule
from utils.create_logger import create_logger
# from utils.load_data import load_gt_roidb, merge_roidb, filter_roidb
from utils.load_data_ucf101 import load_gt_imdb
from utils.load_model import load_param
from utils.PrefetchingIter import PrefetchingIter
from utils.lr_scheduler import WarmupMultiFactorScheduler

def train_net(args, ctx, pretrained, pretrained_flow, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.image_set)
    prefix = os.path.join(final_output_path, prefix)

    # load symbol
    config.symbol = 'resnet_v1_101_flownet_rfcn_ucf101'
    shutil.copy2(os.path.join(curr_path, 'symbols', config.symbol + '.py'), final_output_path)
    sym_instance = eval(config.symbol + '.' + config.symbol)()
    sym = sym_instance.get_train_symbol(config)
    
    feat_sym=sym.get_internals()['cam_softmax_output']

    # setup multi-gpu
    batch_size = len(ctx)
    input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size

    # print config
    pprint.pprint(config)
    logger.info('training config:{}\n'.format(pprint.pformat(config)))

    # load dataset and prepare imdb for training
    config.dataset.dataset = 'UCF101'
    config.dataset.root_path = '/data/weik/data/'
    config.dataset.dataset_path = '/data_ssd2/datasets/UCF101/JPG/'
    config.dataset.traintestlist_path = '/data_ssd2/datasets/UCF101/ucfTrainTestList/'
    config.TRAIN.FLIP = False
    split = '01'
    gtdb = load_gt_imdb(config.dataset.dataset, config.dataset.root_path, config.dataset.dataset_path,
                        config.dataset.traintestlist_path, split=split, flip=config.TRAIN.FLIP)

    # load training data
    train_data = TrainLoader(feat_sym, gtdb, config, batch_size=64, shuffle=True, ctx=ctx, aspect_grouping=True)

    data_shape_dict = dict(train_data.provide_data_single + train_data.provide_label_single)
    pprint.pprint(data_shape_dict)
    sym_instance.infer_shape(data_shape_dict)
    # load and initialize params
    if config.TRAIN.RESUME:
        print('continue training from ', begin_epoch)
        arg_params, aux_params = load_param(prefix, begin_epoch, convert=True)
    else:
        arg_params, aux_params = load_param(pretrained, epoch, convert=True)
        arg_params_flow, aux_params_flow = load_param(pretrained_flow, epoch, convert=True)
        arg_params.update(arg_params_flow)
        aux_params.update(aux_params_flow)
        sym_instance.init_weight(config, arg_params, aux_params)

    # check parameter shapes
    sym_instance.check_parameter_shapes(arg_params, aux_params, data_shape_dict)

    # create solver
    fixed_param_prefix = config.network.FIXED_PARAMS
    data_name = [k[0] for k in train_data.provide_data_single]
    label_name = [k[0] for k in train_data.provide_label_single]
    #module

    mod = mx.mod.Module(symbol=net,
                    context=ctx,
                    data_names=data_name,
                    label_names=label_name)


   
    #mod = MutableModule(sym, data_names=data_names, label_names=label_names,
    #                    logger=logger, context=ctx, max_data_shapes=[train_data.provide_data_single],
    #                    max_label_shapes=[train_data.provide_label_single], fixed_param_prefix=fixed_param_prefix)
 
   
    # allocate memory given the input data and label shapes
    mod.bind(data_shapes=train_data.provide_data_single, label_shapes=train_data.provide_label_single)


    if config.TRAIN.RESUME:
        mod._preload_opt_states = '%s-%04d.states' % (prefix, begin_epoch)

    # decide training params
    # metric
    eval_metrics = mx.metric.CompositeEvalMetric()
    loss_metric = mx.metric.Loss()
    accuracy = mx.metric.Accuracy()
    for evl in [loss_metric,accuracy]:
       eval_metrics.add(evl)
    # callback
    batch_end_callback = mx.callback.Speedometer(train_data.batch_size, frequent=args.frequent)
    epoch_end_callback = mx.callback.module_checkpoint(mod, prefix, period=1, save_optimizer_states=True)
    # decide learning rate
    base_lr = lr
    lr_factor = config.TRAIN.lr_factor
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(gtdb) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr,
                                              config.TRAIN.warmup_step)
    # optimizer
    optimizer_params = {'momentum': config.TRAIN.momentum,
                        'wd': config.TRAIN.wd,
                        'learning_rate': lr,
                        'lr_scheduler': lr_scheduler,
                        'rescale_grad': 1.0,
                        'clip_gradient': None}

    #if not isinstance(train_data, PrefetchingIter):
     #   train_data = PrefetchingIter(train_data)

    
    # initialize parameters by uniform random numbers
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    # use SGD with learning rate 0.1 to train
   # mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
    mod.fit(train_data,
                epoch_end_callback=epoch_end_callback,
                batch_end_callback=batch_end_callback,
                optimizer='sgd',
                optimizer_params=optimizer_params,
                eval_metric=eval_metrics,
                num_epoch=2, 
                begin_epoch=0,
                arg_params=arg_params, 
                aux_params=aux_params)

    # train

   # mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
     #       batch_end_callback=batch_end_callback, kvstore=config.default.kvstore,
     #       optimizer='sgd', optimizer_params=optimizer_params, 
     #       arg_params=arg_params, aux_params=aux_params, begin_epoch=begin_epoch, num_epoch=end_epoch)


def main():
    print('Called with argument:', args)
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(args, ctx, config.network.pretrained, config.network.pretrained_flow, config.network.pretrained_epoch,
              config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr, config.TRAIN.lr_step)


if __name__ == '__main__':
    main()
