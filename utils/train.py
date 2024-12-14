from __future__ import print_function

SEED = 42

import random as rn
rn.seed(SEED)

import numpy as np
np.random.seed(SEED)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(SEED)

import os
import argparse
from BigEarthNet import BigEarthNet
from utils import get_metrics
import json
import importlib
import mlflow
#import mlflow.tensorflow
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_model(args):
    # Initialize MLflow run
    with mlflow.start_run() as run:

    # Log hyperparameters
        for key, value in args.items():
            mlflow.log_param(key, value)

        with tf.Session() as sess:
            # Initialize training and validation iterators
            train_iterator = BigEarthNet(
                args['tr_tf_record_files'], 
                args['batch_size'], 
                args['nb_epoch'], 
                args['shuffle_buffer_size'],
                args['label_type']
            ).batch_iterator
            val_iterator = BigEarthNet(
                args['val_tf_record_files'], 
                args['batch_size'], 
                1, 
                0, 
                args['label_type']
            ).batch_iterator
            
            train_iterator_init_op = train_iterator.initializer
            val_iterator_init_op = val_iterator.initializer
            train_iterator_ins = train_iterator.get_next()
            val_iterator_ins = val_iterator.get_next()

            # Total iterations per epoch
            nb_iteration = int(np.ceil(float(args['training_size']) / args['batch_size']))
            print(f'Number of iterations per epoch: {nb_iteration}')

            # Import model and create network
            model = importlib.import_module('models.' + args['model_name']).DNN_model(args['label_type'], args['modality'])

            #weight_decay = args.get('weight_decay', 0.0001)
            model.create_network()
            
            loss = model.define_loss()
            #l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name]) * weight_decay
            #total_loss = loss + l2_loss
            # Learning rate scheduler
            global_step = tf.Variable(0, trainable=False, name="global_step")
            learning_rate = tf.train.exponential_decay(
                args['learning_rate'], global_step, decay_steps=10000, decay_rate=0.96, staircase=True
            )
            
            # Optimizer with learning rate schedule
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

            # Variable saver
            variables_to_save = tf.global_variables()
            _, metric_means, metric_update_ops = get_metrics(model.multi_hot_label, model.predictions, model.probabilities)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            model_saver = tf.train.Saver(max_to_keep=0, var_list=variables_to_save)
            iteration_idx = 0

            # Fine-tuning (if applicable)
            if args['fine_tune']:
                model_saver.restore(sess, args['model_file'])
                if 'iteration' in args['model_file']:
                    iteration_idx = int(args['model_file'].split('iteration-')[-1])
                print(f'Fine-tuning from iteration: {iteration_idx}')

            # Summaries
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(os.path.join(args['out_dir'], 'logs', 'training'), sess.graph)
            val_summary_writer = tf.summary.FileWriter(os.path.join(args['out_dir'], 'logs', 'validation'), sess.graph)

            mlflow.log_params(args)  # Log hyperparameters
            #mlflow.tensorflow.autolog() 
            # Initialize training
            for epoch in range(args['nb_epoch']):
                print(f'Starting Epoch {epoch + 1}')
                sess.run(train_iterator_init_op)
                epoch_loss = 0.0
                epoch_steps = 0
                progress_bar = tf.keras.utils.Progbar(target=nb_iteration)

                # Training loop
                while True:
                    try:
                        batch_dict = sess.run(train_iterator_ins)
                        _, _, batch_loss, batch_summary = sess.run([train_op, metric_update_ops, loss, summary_op], 
                                                            feed_dict=model.feed_dict(batch_dict, is_training=True))
                        
                        iteration_idx += 1
                        epoch_loss += batch_loss
                        epoch_steps += 1

                        # Log batch loss
                        #batch_loss = batch_loss.item() if isinstance(batch_loss, np.ndarray) else batch_loss

                            # Print and log batch loss for comparison
                        # print(f"Iteration {iteration_idx} - Batch Loss: {batch_loss}")
                        # if iteration_idx % 10 == 0:  # Log every 10 iterations
                        #     mlflow.log_metric('batch_loss', batch_loss, step=iteration_idx)

                        summary_writer.add_summary(batch_summary, iteration_idx)
                        progress_bar.update(iteration_idx % nb_iteration, values=[('loss', batch_loss)]) 

                        # if iteration_idx % 10 == 0:  # Log every 10 iterations
                        #     mlflow.log_metric('batch_loss', batch_loss, step=iteration_idx)

                    except tf.errors.OutOfRangeError:
                        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
                        print(f'Epoch {epoch + 1}/{args["nb_epoch"]} - Training Loss: {avg_epoch_loss:.4f}')

                        # Log epoch loss
                        mlflow.log_metric('train_loss', avg_epoch_loss, step=epoch)
                        break

                    # Validation after every 'val_interval' iterations
                    if iteration_idx % args.get('val_interval', nb_iteration) == 0:
                        # Validation loop
                        sess.run(val_iterator_init_op)
                        val_losses = []
                        while True:
                            try:
                                val_batch_dict = sess.run(val_iterator_ins)
                                _, val_loss = sess.run([metric_update_ops, loss], 
                                                    feed_dict=model.feed_dict(val_batch_dict, is_training=False))
                                val_losses.append(val_loss)
                            except tf.errors.OutOfRangeError:
                                avg_val_loss = np.mean(val_losses) if val_losses else 0
                                print(f'Epoch {epoch + 1}/{args["nb_epoch"]} - Validation Loss: {avg_val_loss:.4f}')
                                
                                # Log validation loss
                                mlflow.log_metric('val_loss', avg_val_loss, step=epoch)
                                
                                # Save validation loss summary
                                val_loss_summary = tf.Summary(value=[tf.Summary.Value(tag='val_loss', simple_value=avg_val_loss)])
                                val_summary_writer.add_summary(val_loss_summary, iteration_idx)
                                break

                    # Save model periodically
                    if (iteration_idx % args['save_checkpoint_per_iteration'] == 0) and (iteration_idx >= args['save_checkpoint_after_iteration']):
                        checkpoint_path = os.path.join(args['out_dir'], 'models', 'iteration')
                        model_saver.save(sess, checkpoint_path, iteration_idx)
                        
                        # Log model checkpoint as an artifact
                        #mlflow.log_artifact(checkpoint_path)

                # Final model save
                final_checkpoint_path = os.path.join(args['out_dir'], 'models', 'iteration')
                model_saver.save(sess, final_checkpoint_path, iteration_idx)
                
                # Log final model as an artifact
                #mlflow.log_artifact(final_checkpoint_path)

        # End the MLflow run
        mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('configs', help='json config file')
    parser_args = parser.parse_args()

    with open(os.path.realpath(parser_args.configs), 'rb') as f:
        model_args = json.load(f)

    run_model(model_args)

# from __future__ import print_function

# SEED = 42

# import random as rn
# rn.seed(SEED)

# import numpy as np
# np.random.seed(SEED)

# import tensorflow as tf
# tf.config.run_functions_eagerly(True)
# # tf.disable_v2_behavior()
# # tf.set_random_seed(SEED)

# import os
# import argparse
# from BigEarthNet import BigEarthNet
# from utils import get_metrics
# import json
# import importlib
# import mlflow

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# def run_model(args):
#     # Initialize MLflow run
#     with mlflow.start_run() as run:
#         # Log hyperparameters
#         for key, value in args.items():
#             mlflow.log_param(key, value)

#         # MirroredStrategy for distributed training
#         strategy = tf.distribute.MirroredStrategy()
#         print(f"Number of devices: {strategy.num_replicas_in_sync}")

#         # Use the strategy scope to define the model and optimizer
#         with strategy.scope():
#             global_step = tf.Variable(0, trainable=False, name="global_step")

#             # Initialize the model
#             model = importlib.import_module('models.' + args['model_name']).DNN_model(args['label_type'], args['modality'])
#             model.create_network()
#             loss = model.define_loss()
#             optimizer = tf.keras.optimizers.Adam(learning_rate=args['learning_rate'])

#             # Define train step function
#             @tf.function
#             def train_step(batch_dict):
#                 with tf.GradientTape() as tape:
#                     predictions = model(batch_dict['image'])
#                     loss_value = loss(predictions, batch_dict[args['label_type'] + '_labels_multi_hot'])
#                 gradients = tape.gradient(loss_value, model.trainable_variables)
#                 optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#                 return loss_value

#             # Training loop
#             for epoch in range(args['nb_epoch']):
#                 print(f'Starting Epoch {epoch + 1}')

#                 # Initialize datasets using BigEarthNet
#                 train_dataset = BigEarthNet(
#                     args['tr_tf_record_files'], 
#                     args['batch_size'], 
#                     args['nb_epoch'], 
#                     args['shuffle_buffer_size'],
#                     args['label_type']
#                 ).dataset
#                 val_dataset = BigEarthNet(
#                     args['val_tf_record_files'], 
#                     args['batch_size'], 
#                     1, 
#                     0, 
#                     args['label_type']
#                 ).dataset

#                 # Initialize iterators for training and validation datasets
#                 train_iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)
#                 val_iterator = tf.compat.v1.data.make_initializable_iterator(val_dataset)

#                 # Initialize the iterators at the beginning of each epoch
#                 # tf.compat.v1.get_default_session().run(train_iterator.initializer)
#                 # tf.compat.v1.get_default_session().run(val_iterator.initializer)

#                 epoch_loss = []
#                 while True:
#                     try:
#                         # Get the next batch from the train iterator
#                         batch_dict = train_iterator.get_next()  # This fetches the next batch
#                         loss_value = strategy.run(train_step, args=(batch_dict,))
#                         epoch_loss.append(loss_value.numpy())
#                     except tf.errors.OutOfRangeError:
#                         break  # End of the epoch

#                 avg_epoch_loss = np.mean(epoch_loss)
#                 print(f'Epoch {epoch + 1}/{args["nb_epoch"]} - Training Loss: {avg_epoch_loss:.4f}')
#                 mlflow.log_metric('train_loss', avg_epoch_loss, step=epoch)

#                 # Validation loop
#                 val_losses = []
#                 while True:
#                     try:
#                         val_batch_dict = val_iterator.get_next()
#                         predictions = model(val_batch_dict['image'])
#                         val_loss = loss(predictions, val_batch_dict[args['label_type'] + '_labels_multi_hot'])
#                         val_losses.append(val_loss.numpy())
#                     except tf.errors.OutOfRangeError:
#                         break  # End of the validation set

#                 avg_val_loss = np.mean(val_losses)
#                 print(f'Epoch {epoch + 1}/{args["nb_epoch"]} - Validation Loss: {avg_val_loss:.4f}')
#                 mlflow.log_metric('val_loss', avg_val_loss, step=epoch)

#                 # Save model periodically
#                 if (epoch + 1) % args.get('save_checkpoint_per_epoch', 1) == 0:
#                     checkpoint_path = os.path.join(args['out_dir'], 'models', f'epoch_{epoch + 1}')
#                     model.save(checkpoint_path)

#         # End the MLflow run
#         mlflow.end_run()
        
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Training script')
#     parser.add_argument('configs', help='json config file')
#     parser_args = parser.parse_args()

#     with open(os.path.realpath(parser_args.configs), 'rb') as f:
#         model_args = json.load(f)

#     run_model(model_args)
