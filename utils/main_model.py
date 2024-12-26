# -*- coding: utf-8 -*-
#
# Main model to define initial placeholders, feed dictionary, and image normalization. 
#
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 27 Feb 2021
# Version: 1.1.1

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from nets.resnet_utils import resnet_arg_scope
from nets.resnet_v1 import resnet_v1_50
from BigEarthNet import BAND_STATS
from utils import sparse_to_dense

class Model:
    def __init__(self, label_type, modality):
        self.label_type = label_type
        self.modality = modality
        self.prediction_threshold = 0.5
        self.is_training = tf.compat.v1.placeholder(tf.bool, [])
        self.nb_class = 19 if label_type == 'BigEarthNet-19' else 43

        # self.B02 = tf.compat.v1.placeholder(tf.float32, [None, 120, 120], name='B02')
        # self.B03 = tf.compat.v1.placeholder(tf.float32, [None, 120, 120], name='B03')
        # self.B04 = tf.compat.v1.placeholder(tf.float32, [None, 120, 120], name='B04')

        #self.image = tf.compat.v1.placeholder(tf.float32, [None, 480, 480, 3], name='image')
        #self.image = tf.compat.v1.placeholder(tf.float32, [None, 448, 448, 3], name='image')
        self.image = tf.compat.v1.placeholder(tf.float32, [None, 120, 120, 3], name='image')

        #self.bands_10m = tf.stack([self.B04, self.B03, self.B02], axis=3)

        
        if self.modality == 'S1':
            self.img = self.S1_img
        elif self.modality == 'S2':
            #self.img =  self.bands_10m
            self.img =  self.image
        elif self.modality == 'MM':
            self.img = self.S12_img
        
        self.multi_hot_label = tf.compat.v1.placeholder(tf.float32, shape=(None, self.nb_class))
        self.model_path = tf.compat.v1.placeholder(tf.string)
        #self.patch_name_s2 = tf.compat.v1.placeholder(tf.string)

    def feed_dict(self, batch_dict, is_training=False, model_path=''):
        # image = ((batch_dict['image'] - BAND_STATS['S2']['mean']['rgb']) / BAND_STATS['S2']['std']['rgb']).astype(np.float32)
   
        image = (batch_dict['image'] / 255.0).astype(np.float32)

        # B02  = ((batch_dict['B02'] - BAND_STATS['S2']['mean']['B02']) / BAND_STATS['S2']['std']['B02']).astype(np.float32)
        # B03  = ((batch_dict['B03'] - BAND_STATS['S2']['mean']['B03']) / BAND_STATS['S2']['std']['B03']).astype(np.float32)
        # B04  = ((batch_dict['B04'] - BAND_STATS['S2']['mean']['B04']) / BAND_STATS['S2']['std']['B04']).astype(np.float32)

        multi_hot_label = batch_dict[
                'original_labels_multi_hot'
            ].astype(np.float32) if self.label_type == 'original' else batch_dict[
                'BigEarthNet-19_labels_multi_hot'
            ].astype(np.float32)

        # Label and patch names can be read in the following way:
        #
        # original_labels = sparse_to_dense(batch_dict['original_labels'].indices, batch_dict['original_labels'].values)
        # BigEarthNet-19_labels = sparse_to_dense(batch_dict['BigEarthNet-19_labels'].indices, batch_dict['BigEarthNet-19_labels'].values)
        # patch_name_s1 = sparse_to_dense(batch_dict['patch_name_s1'].indices, batch_dict['patch_name_s1'].values)
        patch_name_s2 = sparse_to_dense(batch_dict['patch_name'].indices, batch_dict['patch_name'].values)
        #print('patch',patch_name_s2)
        
        return {
                # self.B02: B02,
                # self.B03: B03,
                # self.B04: B04,
                self.image: image,
                self.multi_hot_label: multi_hot_label, 
                self.is_training:is_training,
                self.model_path:model_path,
                #patch_name_s2: patch_name_s2
            }

    def define_loss(self):
        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.multi_hot_label, logits=self.logits)
        tf.summary.scalar('sigmoid_cross_entropy_loss', self.loss)
        return self.loss