import tensorflow.compat.v1 as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))


import numpy as np
import sys 
sys.path.append('/netscratch/mudraje/super_resolution_remote_sensing/bigearthnet-models-tf')
from nets import resnet_v1
import tf_slim as slim

slim = slim
import gradcam_utils


img1 = gradcam_utils.load_image("/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb/images/S2B_MSIL2A_20180421T114349_83_24.jpg", normalize=True)

batch_img = img1.reshape((1, 224, 224, 3))
batch_label = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
batch_label = batch_label.reshape(1, -1)
batch_size = 1

eval_graph = tf.Graph()
with eval_graph.as_default():
    with eval_graph.gradient_override_map({'Relu': 'GuidedRelu'}):
        images = tf.placeholder("float", [batch_size, 224, 224, 3])
        labels = tf.placeholder(tf.float32, [batch_size, 19])
        
        preprocessed_images = gradcam_utils.resnet_preprocess(images)
        
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            with slim.arg_scope([slim.batch_norm], is_training=False):
                # is_training=False means batch-norm is not in training mode. Fixing batch norm layer.
                # net is logit for resnet_v1. See is_training messing up issue: https://github.com/tensorflow/tensorflow/issues/4887
                net, end_points = resnet_v1.resnet_v1_152(preprocessed_images, 19)
        prob = end_points['predictions']
        print(prob)
        
        cost = (-1) * tf.reduce_sum(tf.multiply(labels, tf.log(prob)), axis=1)
        #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=net))
        print('cost:', cost)
        y_c = tf.reduce_sum(tf.multiply(net, labels), axis=1)
        print('y_c:', y_c)

        # Get last convolutional layer gradient for generating gradCAM visualization
        # print('endpoints:', end_points.keys())
        target_conv_layer = end_points['resnet_v1_152/block4/unit_2/bottleneck_v1']
        target_conv_layer_grad = tf.gradients(y_c, target_conv_layer)[0]

        # Guided backpropagtion back to input layer
        gb_grad = tf.gradients(cost, images)[0]

        init = tf.global_variables_initializer()
        
        # Load resnet v1 weights
        
        # latest_checkpoint = tf.train.latest_checkpoint("model/resnet_v1_50.ckpt")
        latest_checkpoint = "/netscratch/mudraje/super_resolution_remote_sensing/checkpoints/Resnet152_normal_32batch_normalization/models/iteration-252840"

        reader = tf.train.NewCheckpointReader(latest_checkpoint)
        saved_shapes = reader.get_variable_to_shape_map()
        variables_to_restore = tf.global_variables()
        for var in variables_to_restore:
          if not var.name.split(':')[0] in saved_shapes:
            print("WARNING. Saved weight not exists in checkpoint. Init var:", var.name)
          else:
            # print("Load saved weight:", var.name)
            pass

        var_names = sorted([(var.name, var.name.split(':')[0]) for var in variables_to_restore
                if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                try:
                    curr_var = tf.get_variable(saved_var_name)
                    var_shape = curr_var.get_shape().as_list()
                    if var_shape == saved_shapes[saved_var_name]:
                        # print("restore var:", saved_var_name)
                        restore_vars.append(curr_var)
                except ValueError:
                    print("Ignore due to ValueError on getting var:", saved_var_name) 
        saver = tf.train.Saver(restore_vars)
        
        
        
# Run tensorflow 

with tf.Session(graph=eval_graph) as sess:    
    sess.run(init)    
    # sess.run(tf.local_variables_initializer())
    saver.restore(sess, latest_checkpoint)
    
    prob = sess.run(prob, feed_dict={images: batch_img})
    print('prob', prob)
    
    # gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad], feed_dict={images: batch_img, labels: prob})
    gb_grad_value, target_conv_layer_value, target_conv_layer_grad_value = sess.run([gb_grad, target_conv_layer, target_conv_layer_grad], feed_dict={images: batch_img, labels: batch_label})    
    
    for i in range(batch_size):
        # print('See visualization of below category')
        # utils.print_prob(batch_label[i], './synset.txt')
        gradcam_utils.print_prob(prob[i], '/netscratch/mudraje/super_resolution_remote_sensing/utils/synset.txt')
        # print('gb_grad_value[i]:', gb_grad_value[i])
        # print('gb_grad_value[i] shape:', gb_grad_value[i].shape)
        save_path = "/netscratch/mudraje/super_resolution_remote_sensing/checkpoints/Resnet152_normal_32batch_normalization/results_cam_83_24_252840"

        gradcam_utils.visualize(batch_img[i], target_conv_layer_value[i], target_conv_layer_grad_value[i], gb_grad_value[i], save_path)