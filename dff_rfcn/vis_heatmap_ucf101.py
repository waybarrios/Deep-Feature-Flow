import _init_paths

import argparse
import os
import glob
import sys
import logging
import pprint
import cv2
from config.config import config, update_config
from utils.image import resize, transform
import numpy as np
# get config
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/dff_rfcn/cfgs/dff_rfcn_vid_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet', config.MXNET_VERSION))
import mxnet as mx
from core.tester import im_detect, Predictor
from symbols import *
import pandas as pd
import numpy as np

from utils.load_model import load_param
from utils.show_boxes import show_boxes, draw_boxes, draw_heatmap
from utils.tictoc import tic, toc



def parse_args():
    parser = argparse.ArgumentParser(description='Show Deep Feature Flow demo')
    args = parser.parse_args()
    return args

args = parse_args()


# load class names
def load_labels(label_csv_path):
    data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
    labels = []
    for i in range(data.shape[0]):
        labels.append(data.ix[i, 1])
    return labels

def main():
    # get symbol
    pprint.pprint(config)
    config.symbol = 'resnet_v1_101_flownet_rfcn_ucf101'
    model = '/home/weik/Documents/Deep-Feature-Flow/output/dff_rfcn/imagenet_vid/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem/DET_train_30classes_VID_train_15frames/dff_rfcn_vid'
    sym_instance = eval(config.symbol + '.' + config.symbol)()

    vis_sym = sym_instance.get_cam_test_symbol(config)

    # set up class names
    traintestlist_path = '/data_ssd2/datasets/UCF101/ucfTrainTestList/'
    classes = load_labels(os.path.join(traintestlist_path, 'classInd.txt'))
    num_classes = len(classes)

    # load demo data
    image_names = glob.glob('/data_ssd2/datasets/UCF101/JPG/PlayingSitar/v_PlayingSitar_g03_c05/*.jpg')
    output_dir = cur_path + '/../demo/ucf101/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #
    data = []
    key_im_tensor = None
    for idx, im_name in enumerate(image_names):
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = config.SCALES[0][0]
        max_size = config.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=config.network.IMAGE_STRIDE)
        im_tensor = transform(im, config.network.PIXEL_MEANS)

        data.append({'data': im_tensor})

    # get predictor
    data_names = ['data']
    label_names = []
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]

    max_data_shape = [('data', (1, 3, 240, 320))]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for i in xrange(len(data))]
    arg_params, aux_params = load_param(model, 2, process=True)
    weight = arg_params['cam_fc_weights']
    key_predictor = Predictor(vis_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)

    # test
    time = 0
    count = 0
    for idx, im_name in enumerate(image_names):
        data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                     provide_label=[None])
        cam_resnet, conv_3x3 = key_predictor.predict(data_batch)

        # visualize
        im = cv2.imread(im_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        heat_map = heat_map_generate(conv_3x3, weight)

        # show_heatmap
        out_im = draw_heatmap(im, heat_map)
        _, filename = os.path.split(im_name)
        cv2.imwrite(output_dir + filename,out_im)

    print 'done'


def heat_map_generate(conv_3x3, weight):
    feature_map = np.asnumpy(conv_3x3)
    weight = np.asnumpy(weight)

    heat_map = np.average(feature_map, axis=2, weights=weight)

    return heat_map



if __name__ == '__main__':
    main()