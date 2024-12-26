

import skimage
import skimage.io
import skimage.transform
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg
from matplotlib import colormaps as cm
from PIL.Image import fromarray
import tensorflow as tf
from typing import cast
from PIL import Image
import os

from skimage import io
from skimage.transform import resize

import cv2

# synset = [l.strip() for l in open('synset.txt').readlines()]

def resnet_preprocess(resized_inputs):
    """Faster R-CNN Resnet V1 preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

    Args:
      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
        representing a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: A [batch, height_out, width_out, channels] float32
        tensor representing a batch of images.
    """
    channel_means = tf.constant([123.68, 116.779, 103.939],
        dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
    return resized_inputs - channel_means


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, normalize=True):
    """
    args:
        normalize: set True to get pixel value of 0~1
    """
    # load image
    img = skimage.io.imread(path)
    if normalize:
        img = img / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()


    # print "Original Image Shape: ", img.shape
    # we crop image from center
    # short_edge = min(img.shape[:2])
    # yy = int((img.shape[0] - short_edge) / 2)
    # xx = int((img.shape[1] - short_edge) / 2)
    # crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(img, (224, 224), preserve_range=True) # do not normalize at transform. 
    return resized_img

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    print(synset)
    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1, prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print("Top5: ", top5)
    return top1



def visualize(image, conv_output, conv_grad, gb_viz, save_path):
    output = conv_output           # [7,7,512]
    grads_val = conv_grad          # [7,7,512]
    print("grads_val shape:", grads_val.shape)
    print("gb_viz shape:", gb_viz.shape)

    weights = np.mean(grads_val, axis = (0, 1)) # alpha_k, [512]
    cam = np.zeros(output.shape[0 : 2], dtype = np.float32)	# [7,7]
    

    # Taking a weighted average
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    # Passing through ReLU
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) # scale 0 to 1.0
    cam = resize(cam, (224,224), preserve_range=True)

    img = image.astype(float)
    img -= np.min(img)
    img /= img.max()
    # print(img)
    cam_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
    # cam = np.float32(cam) + np.float32(img)
    # cam = 255 * cam / np.max(cam)
    # cam = np.uint8(cam)
              
    
    # fig = plt.figure()    
    # ax = fig.add_subplot(111)
    # imgplot = plt.imshow(img)
    # ax.set_title('Input Image')
    
    # fig = plt.figure(figsize=(12, 16))    
    # ax = fig.add_subplot(131)
    # imgplot = plt.imshow(cam_heatmap)
    # ax.set_title('Grad-CAM')    
    
    gb_viz = np.dstack((
            gb_viz[:, :, 0],
            gb_viz[:, :, 1],
            gb_viz[:, :, 2],
        ))       
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    # ax = fig.add_subplot(132)
    # imgplot = plt.imshow(gb_viz)
    # ax.set_title('guided backpropagation')
    

    gd_gb = np.dstack((
            gb_viz[:, :, 0] * cam,
            gb_viz[:, :, 1] * cam,
            gb_viz[:, :, 2] * cam,
        ))            
    # ax = fig.add_subplot(133)
    # imgplot = plt.imshow(gd_gb)
    # ax.set_title('guided Grad-CAM')


    os.makedirs(save_path, exist_ok=True)

    # Save the Input Image
    input_image_path = os.path.join(save_path, "input_image.png")
    plt.imsave(input_image_path, img)
    print(f"Input Image saved at: {input_image_path}")

    # Save Grad-CAM Heatmap
    grad_cam_path = os.path.join(save_path, "grad_cam.png")
    plt.imsave(grad_cam_path, cam_heatmap)
    print(f"Grad-CAM Heatmap saved at: {grad_cam_path}")

    # Save Guided Backpropagation Visualization
    gb_viz_path = os.path.join(save_path, "guided_backpropagation.png")
    plt.imsave(gb_viz_path, gb_viz)
    print(f"Guided Backpropagation saved at: {gb_viz_path}")

    # Save Guided Grad-CAM
    gd_gb_path = os.path.join(save_path, "guided_grad_cam.png")
    plt.imsave(gd_gb_path, gd_gb)
    print(f"Guided Grad-CAM saved at: {gd_gb_path}")


    #colormap=cv2.COLORMAP_VIRIDIS
    # plt.show()
    alpha=0.7
    img = (255 * img).astype(np.uint8)
    # # heatmap = cv2.applyColorMap(cam_heatmap, colormap)
    #output = cv2.addWeighted(img, alpha, cam_heatmap, 1 - alpha, 0)
    #colormap = "jet"
    #cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    #overlay = mask.resize(img.size, resample=Resampling.BICUBIC)
    #overlay = (255 * cmap(np.asarray(cam) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    #output = fromarray((alpha * np.asarray(img) + (1 - alpha) * cast(np.ndarray, overlay)).astype(np.uint8))
    overlay = cv2.addWeighted(img, alpha, cam_heatmap, 1 - alpha, 0)
    
    # Convert the overlay to a PIL Image for saving
    #overlay_img = Image.fromarray(overlay)

    overlay_grad_cam_path = os.path.join(save_path, "overlay_grad_cam.png")
    plt.imsave(overlay_grad_cam_path, overlay)
    print(f"Grad-CAM Heatmap saved at: {overlay_grad_cam_path}")