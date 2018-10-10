# --------------------------------------------------------
# Deep Feature Flow
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xizhou Zhu
# --------------------------------------------------------

"""
ImageNet VID database
This class loads ground truth notations from standard ImageNet VID XML data formats
and transform them into IMDB format. Selective search is used for proposals, see roidb
function. Results are written as the ImageNet VID format. Evaluation is based on mAP
criterion.
"""

import cPickle
import pandas as pd
import cv2
import os
import numpy as np
from PIL import Image

from imdb_ucf101 import IMDB
from imagenet_vid_eval import vid_eval
from ds_utils import unique_boxes, filter_small_boxes


class UCF101(IMDB):
    def __init__(self, root_path, dataset_path, traintestlist_path, result_path=None, split = '01'):
        """
        fill basic information to initialize imdb
        """
        super(UCF101, self).__init__('UCF101', root_path, dataset_path, traintestlist_path,
                                          result_path, split)  # set self.name

        self.root_path = root_path
        self.data_path = dataset_path
        self.traintestlist_path = traintestlist_path
        self.split = split

        self.classes = self.load_labels(os.path.join(traintestlist_path, 'classInd.txt'))

        self.num_classes = len(self.classes)
        self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        print 'num_images', self.num_images

    # load class names
    def load_labels(self, label_csv_path):
        data = pd.read_csv(label_csv_path, delimiter=' ', header=None)
        labels = []
        for i in range(data.shape[0]):
            labels.append(data.iloc[i, data.columns.get_loc(1)])
        return labels


    def load_value_file(self, file_path):
        with open(file_path, 'r') as input_file:
            value = float(input_file.read().rstrip('\n\r'))

        return value

    def load_image_set_index(self):
        """
        find out which indexes correspond to given image set (train or val)
        :return:
        """

        image_set_list_file = os.path.join(self.traintestlist_path, 'trainlist' + self.split + '.txt')
        assert os.path.exists(image_set_list_file), 'Path does not exist: {}'.format(image_set_list_file)


        with open(image_set_list_file) as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
        if len(lines[0]) == 2:
            self.image_set_list = [x[0].split('.')[0] for x in lines]
            self.image_set_list_lable = [x[1] for x in lines]

            image_set_index = []
            frame_id = []
            image_set_label = []
            image_set_list_start_end = []
            count = 0
            for video_idx, video_name in enumerate(self.image_set_list):
                n_frames_file_path = os.path.join(self.data_path, video_name, 'n_frames')
                n_frames = int(self.load_value_file(n_frames_file_path))

                for frame_idx in np.arange(1, n_frames+1):
                    image_set_index.append(os.path.join(video_name, "image_{:05d}.jpg".format(frame_idx)))
                    frame_id.append(count+frame_idx-1)
                    image_set_label.append(self.image_set_list_lable[video_idx])

                image_set_list_start_end.append([count, count+n_frames-1])
                count = count + n_frames

            self.image_set_index = image_set_index
            self.frame_id = frame_id
            self.image_set_label = image_set_label
            self.image_set_list_start_end = image_set_list_start_end


    def image_path_from_index(self, index):
        """
        given image index, find out full path
        :param index: index of a specific image
        :return: full path of this image
        """
        image_file = os.path.join(self.data_path, index)

        return image_file

    def load_vid_annotation(self, iindex):
        """
        for a given index, load image and bounding boxes info from XML file
        :param index: index of a specific image
        :return: record['iamge', 'frame_id', 'label', 'flipped', 'width', 'height']
        """
        index = self.image_set_index[iindex]

        roi_rec = dict()
        roi_rec['image'] = self.image_path_from_index(index)
        roi_rec['frame_id'] = self.frame_id[iindex]
        roi_rec['label'] = self.image_set_label[iindex]

        img = Image.open(roi_rec['image'])

        roi_rec['height'] = float(img.height)
        roi_rec['width'] = float(img.width)

        roi_rec['flipped'] = False

        return roi_rec

    def gt_db(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_db.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt db loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_db = [self.load_vid_annotation(index) for index in range(0, len(self.image_set_index))]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_db, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_db
