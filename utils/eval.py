# -*- coding: utf-8 -*-
#
# This script can be used to evaluate the performance of a deep learning model, pre-trained on the BigEarthNet.
#
# To run the code, you need to provide the json file which was used for training before. 
# 
# Author: Gencer Sumbul, http://www.user.tu-berlin.de/gencersumbul/
# Email: gencer.suembuel@tu-berlin.de
# Date: 23 Dec 2019
# Version: 1.0.1
# Usage: eval.py [CONFIG_FILE_PATH]

from __future__ import print_function
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import subprocess, time, os
import argparse
from BigEarthNet import BigEarthNet
from utils import get_metrics
import json
import importlib
from utils import sparse_to_dense

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def eval_model(args):
#     print(args['model_file'])
#     with tf.Session() as sess:
#         iterator = BigEarthNet(
#             args['test_tf_record_files'], 
#             args['batch_size'], 
#             1, 
#             0,
#             args['label_type']
#         ).batch_iterator
#         nb_iteration = int(np.ceil(float(args['test_size']) / args['batch_size']))
#         iterator_ins = iterator.get_next()

#         model = importlib.import_module('models.' + args['model_name']).DNN_model(args['label_type'], args['modality'])
#         model.create_network()
#         loss = model.define_loss()

#         variables_to_restore = tf.global_variables()
#         metric_names, metric_means, metric_update_ops = get_metrics(model.multi_hot_label, model.predictions, model.probabilities)
#         sess.run(tf.global_variables_initializer())
#         sess.run(tf.local_variables_initializer())

#         model_saver = tf.train.Saver(max_to_keep=0, var_list=variables_to_restore)
#         model_file = args['model_file']
#         model_saver.restore(sess, model_file)
        
#         summary_op = tf.summary.merge_all()
#         summary_writer = tf.summary.FileWriter(os.path.join(args['out_dir'], 'logs', 'test'), sess.graph)

#         iteration_idx = 0

#         progress_bar = tf.keras.utils.Progbar(target=nb_iteration)
#         eval_res = {}
#                 # Initialize testing iterator
#         sess.run(iterator.initializer)
#         while True:
#             try:
#                 batch_dict = sess.run(iterator_ins)
#                 iteration_idx += 1
#                 progress_bar.update(iteration_idx)
#             except tf.errors.OutOfRangeError:
#                 print()
#                 means = sess.run(metric_means[0])
#                 for idx, name in enumerate(metric_names[0]):
#                     eval_res[name] = str(means[idx])
#                     print(name, means[idx])
#                 break
            
#             sess_res = sess.run([metric_update_ops, summary_op] + metric_means[1], feed_dict=model.feed_dict(
#                         batch_dict))
#             summary_writer.add_summary(sess_res[1], iteration_idx)
#             metric_means_res = sess_res[2:]
#             #progress_bar.update(iteration_idx, values=[('loss', sess_res[0])])
#         for idx, name in enumerate(metric_names[1]):
#             eval_res[name] = str(metric_means_res[idx])
#             print(name, metric_means_res[idx])

#         # with open(os.path.join(args['out_dir'], 'eval_result.json'), 'wb') as f:
#         #     json.dump(eval_res, f)

#         all_predictions = model.predictions
#         all_true_labels = model.multi_hot_label
#         all_patch_names = model.patch_name_s2

#         results_df = pd.DataFrame({
#             'patch_name': [list(name) for name in all_patch_names],
#             'predictions': [list(pred) for pred in all_predictions],
#             'true_labels': [list(label) for label in all_true_labels]
#         })
        
#         # Save results to CSV
#         results_df.to_csv(os.path.join(args['out_dir'],'results.csv'), index=False)


def eval_model(args):
    print(args['model_file'])

    with tf.Session() as sess:
        iterator = BigEarthNet(
            args['test_tf_record_files'], 
            args['batch_size'], 
            1, 
            0,
            args['label_type']
        ).batch_iterator
        nb_iteration = int(np.ceil(float(args['test_size']) / args['batch_size']))
        iterator_ins = iterator.get_next()

        model = importlib.import_module('models.' + args['model_name']).DNN_model(args['label_type'], args['modality'])
        model.create_network()
        loss = model.define_loss()

        variables_to_restore = tf.global_variables()
        metric_names, metric_means, metric_update_ops = get_metrics(model.multi_hot_label, model.predictions, model.probabilities)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        model_saver = tf.train.Saver(max_to_keep=0, var_list=variables_to_restore)
        model_file = args['model_file']
        model_saver.restore(sess, model_file)
        
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(args['out_dir'], 'logs', 'test'), sess.graph)

        iteration_idx = 0

        progress_bar = tf.keras.utils.Progbar(target=nb_iteration)
        eval_res = {}
        
        # Initialize testing iterator
        sess.run(iterator.initializer)

        # Initialize lists to hold predictions and true labels
        all_predictions = []
        all_true_labels = []
        all_patch_names = []

        while True:
            try:
                batch_dict = sess.run(iterator_ins)
                iteration_idx += 1
                progress_bar.update(iteration_idx)
                
                sess_res = sess.run([metric_update_ops, summary_op] + metric_means[1], feed_dict=model.feed_dict(batch_dict))
                summary_writer.add_summary(sess_res[1], iteration_idx)
                metric_means_res = sess_res[2:]

                predictions = sess.run(model.predictions, feed_dict=model.feed_dict(batch_dict))
                true_labels = batch_dict[args['label_type'] + '_labels_multi_hot']
                patch_names = sparse_to_dense(batch_dict['patch_name'].indices, batch_dict['patch_name'].values)

                all_predictions.extend(predictions)
                all_true_labels.extend(true_labels)
                all_patch_names.extend(patch_names)
                
            except tf.errors.OutOfRangeError:
                print()
                means = sess.run(metric_means[0])
                for idx, name in enumerate(metric_names[0]):
                    eval_res[name] = str(means[idx])
                    print(name, means[idx])
                break
        
        for idx, name in enumerate(metric_names[1]):
            eval_res[name] = str(metric_means_res[idx])
            print(name, metric_means_res[idx])


        results_df = pd.DataFrame({
            'patch_name': [list(name) for name in all_patch_names],
            'predictions': [list(pred) for pred in all_predictions],
            'true_labels': [list(label) for label in all_true_labels]
        })
        # Save results to CSV
        results_df.to_csv(os.path.join(args['out_dir'],'results.csv'), index=False)
        
        # Dump all collected data to JSON

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test arguments')
    parser.add_argument('settings', help='json settings file')

    parser_args = parser.parse_args()

    # with open('configs/base.json', 'rb') as f:
    #     args = json.load(f)

    with open(os.path.realpath(parser_args.settings), 'rb') as f:
        model_args = json.load(f)

    # args.update(model_args)
    eval_model(model_args)
