# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuwen Xiong
# --------------------------------------------------------

import numpy as np
from dataset import *
from dataset.ucf101 import UCF101


def load_gt_imdb(dataset_name, root_path, dataset_path, traintestlist_path, result_path=None, split = '01',
                  flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(root_path, dataset_path, traintestlist_path, result_path, split)
    gtdb = imdb.gt_db()
    if flip:
        gtdb = imdb.append_flipped_images(gtdb)
    return gtdb
