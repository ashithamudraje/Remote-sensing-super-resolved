# # Copyright (C) 2020-2024, François-Guillaume Fernandez.

# # This program is licensed under the Apache License 2.0.
# # See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

# """
# CAM visualization
# """

# import argparse
# import math
# from io import BytesIO

# import matplotlib.pyplot as plt
# import requests
# import torch
# from PIL import Image
# from torchvision import models
# from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor

# from torchcam import methods
# from torchcam.utils import overlay_mask


# def main(args):
#     if args.device is None:
#         args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

#     device = torch.device(args.device)

#     # Pretrained imagenet model
#     model = models.__dict__[args.arch](pretrained=True).to(device=device)
#     # Freeze the model
#     for p in model.parameters():
#         p.requires_grad_(False)

#     # Image
#     img_path = BytesIO(requests.get(args.img, timeout=5).content) if args.img.startswith("http") else args.img
#     pil_img = Image.open(img_path, mode="r").convert("RGB")

#     # Preprocess image
#     img_tensor = normalize(
#         to_tensor(resize(pil_img, (224, 224))),
#         [0.485, 0.456, 0.406],
#         [0.229, 0.224, 0.225],
#     ).to(device=device)
#     img_tensor.requires_grad_(True)

#     if isinstance(args.method, str):
#         cam_methods = args.method.split(",")
#     else:
#         cam_methods = [
#             "CAM",
#             # "GradCAM",
#             # "GradCAMpp",
#             # "SmoothGradCAMpp",
#             # "ScoreCAM",
#             # "SSCAM",
#             # "ISCAM",
#             # "XGradCAM",
#             # "LayerCAM",
#         ]
#     # Hook the corresponding layer in the model
#     cam_extractors = [
#         methods.__dict__[name](model, target_layer=args.target, enable_hooks=False) for name in cam_methods
#     ]

#     # Homogenize number of elements in each row
#     num_cols = math.ceil((len(cam_extractors) + 1) / args.rows)
#     _, axes = plt.subplots(args.rows, num_cols, figsize=(6, 4))
#     # Display input
#     ax = axes[0][0] if args.rows > 1 else axes[0] if num_cols > 1 else axes
#     ax.imshow(pil_img)
#     ax.set_title("Input", size=8)

#     for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors):
#         extractor.enable_hooks()
#         model.zero_grad()
#         scores = model(img_tensor.unsqueeze(0))

#         # Select the class index
#         class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx

#         # Use the hooked data to compute activation map
#         activation_map = extractor(class_idx, scores)[0].squeeze(0).cpu()

#         # Clean data
#         extractor.disable_hooks()
#         extractor.remove_hooks()
#         # Convert it to PIL image
#         # The indexing below means first image in batch
#         heatmap = to_pil_image(activation_map, mode="F")
#         # Plot the result
#         result = overlay_mask(pil_img, heatmap, alpha=args.alpha)

#         ax = axes[idx // num_cols][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes

#         ax.imshow(result)
#         ax.set_title(extractor.__class__.__name__, size=8)

#     # Clear axes
#     if num_cols > 1:
#         for _axes in axes:
#             if args.rows > 1:
#                 for ax in _axes:
#                     ax.axis("off")
#             else:
#                 _axes.axis("off")

#     else:
#         axes.axis("off")

#     plt.tight_layout()
#     if args.savefig:
#         plt.savefig(args.savefig, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
#     plt.show(block=not args.noblock)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Saliency Map comparison",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument("--arch", type=str, default="resnet18", help="Name of the architecture")
#     parser.add_argument(
#         "--img",
#         type=str,
#         default="https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg",
#         help="The image to extract CAM from",
#     )
#     parser.add_argument("--class-idx", type=int, default=232, help="Index of the class to inspect")
#     parser.add_argument(
#         "--device",
#         type=str,
#         default=None,
#         help="Default device to perform computation on",
#     )
#     parser.add_argument("--savefig", type=str, default=None, help="Path to save figure")
#     parser.add_argument("--method", type=str, default=None, help="CAM method to use")
#     parser.add_argument("--target", type=str, default=None, help="the target layer")
#     parser.add_argument("--alpha", type=float, default=0.5, help="Transparency of the heatmap")
#     parser.add_argument("--rows", type=int, default=1, help="Number of rows for the layout")
#     parser.add_argument(
#         "--noblock",
#         dest="noblock",
#         help="Disables blocking visualization",
#         action="store_true",
#     )
#     args = parser.parse_args()

#     main(args)


# import argparse
# import math
# from io import BytesIO

# import matplotlib.pyplot as plt
# import requests
# import tensorflow as tf
# import torch
# from PIL import Image
# from torchvision import models
# from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
# from tensorflow.keras.models import load_model
# from torchcam import methods
# from torchcam.utils import overlay_mask

# def load_custom_model(arch, checkpoint_path, device):
# # Method to load the custom pretrained model
#     model = models.__dict__[arch](pretrained=False)  # We set pretrained=False because we're loading a custom model

#     # Load the weights using tf.train.Checkpoint
#     checkpoint = tf.train.Checkpoint(model=model)
#     status = checkpoint.restore(checkpoint_path)
    
#     # Ensure that the variables are initialized
#     status.assert_existing_objects_matched()

#     # Move model to the specified device
#     model.to(device)

#     return model

# def main(args):
#     # Check device availability
#     if args.device is None:
#         args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

#     device = torch.device(args.device)

#     # Load the custom model
#     model = load_custom_model(args.arch, args.checkpoint, device)
#     model = model.to(device)

#     # Freeze the model
#     for p in model.parameters():
#         p.requires_grad_(False)

#     # Image
#     img_path = BytesIO(requests.get(args.img, timeout=5).content) if args.img.startswith("http") else args.img
#     pil_img = Image.open(img_path, mode="r").convert("RGB")

#     # Preprocess image
#     img_tensor = normalize(
#         to_tensor(resize(pil_img, (480, 480))),
#         [0.485, 0.456, 0.406],
#         [0.229, 0.224, 0.225],
#     ).to(device=device)
#     img_tensor.requires_grad_(True)

#     if isinstance(args.method, str):
#         cam_methods = args.method.split(",")
#     else:
#         cam_methods = [
#             "CAM",
#             # "GradCAM",
#             # "GradCAMpp",
#             # "SmoothGradCAMpp",
#             # "ScoreCAM",
#             # "SSCAM",
#             # "ISCAM",
#             # "XGradCAM",
#             # "LayerCAM",
#         ]
#     # Hook the corresponding layer in the model
#     cam_extractors = [
#         methods.__dict__[name](model, target_layer=args.target, enable_hooks=False) for name in cam_methods
#     ]

#     # Homogenize number of elements in each row
#     num_cols = math.ceil((len(cam_extractors) + 1) / args.rows)
#     _, axes = plt.subplots(args.rows, num_cols, figsize=(6, 4))
#     # Display input
#     ax = axes[0][0] if args.rows > 1 else axes[0] if num_cols > 1 else axes
#     ax.imshow(pil_img)
#     ax.set_title("Input", size=8)

#     for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors):
#         extractor.enable_hooks()
#         model.zero_grad()
#         scores = model(img_tensor.unsqueeze(0))

#         # Select the class index
#         class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx

#         # Use the hooked data to compute activation map
#         activation_map = extractor(class_idx, scores)[0].squeeze(0).cpu()

#         # Clean data
#         extractor.disable_hooks()
#         extractor.remove_hooks()
#         # Convert it to PIL image
#         heatmap = to_pil_image(activation_map, mode="F")
#         # Plot the result
#         result = overlay_mask(pil_img, heatmap, alpha=args.alpha)

#         ax = axes[idx // num_cols][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes

#         ax.imshow(result)
#         ax.set_title(extractor.__class__.__name__, size=8)

#     # Clear axes
#     if num_cols > 1:
#         for _axes in axes:
#             if args.rows > 1:
#                 for ax in _axes:
#                     ax.axis("off")
#             else:
#                 _axes.axis("off")

#     else:
#         axes.axis("off")

#     plt.tight_layout()
#     if args.savefig:
#         plt.savefig(args.savefig, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
#     plt.show(block=not args.noblock)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Saliency Map comparison",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument("--arch", type=str, default="resnet152", help="Name of the architecture")
#     parser.add_argument(
#         "--img",
#         type=str,
#         default="https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg",
#         help="The image to extract CAM from",
#     )
#     parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint")
#     parser.add_argument("--class-idx", type=int, default=232, help="Index of the class to inspect")
#     parser.add_argument("--device", type=str, default=None, help="Default device to perform computation on")
#     parser.add_argument("--savefig", type=str, default=None, help="Path to save figure")
#     parser.add_argument("--method", type=str, default=None, help="CAM method to use")
#     parser.add_argument("--target", type=str, default=None, help="the target layer")
#     parser.add_argument("--alpha", type=float, default=0.5, help="Transparency of the heatmap")
#     parser.add_argument("--rows", type=int, default=1, help="Number of rows for the layout")
#     parser.add_argument(
#         "--noblock",
#         dest="noblock",
#         help="Disables blocking visualization",
#         action="store_true",
#     )
#     args = parser.parse_args()

#     main(args)
from gradcam import GradCAM
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to the model")
ap.add_argument("-wd", "--width", type=int, required=True,
	help="width of the image")
ap.add_argument("-ht", "--height", type=int, required=True,
	help="height of the image")
ap.add_argument("-l", "--layer", type=str, default="None",
	help="gradcam of specific layer")
ap.add_argument("-o", "--output", type=str, default=None, 
    help="Path to save figure")
args = vars(ap.parse_args())

# load the custom model and print summary()
model = load_model(args["model"])
model.summary()

image = args["image"]
w, h = args["width"], args["height"]
orig = cv2.imread(image)
resized = cv2.resize(orig, (w, h))

# load the input image from disk (in Keras/TensorFlow format) and
# preprocess it
image = load_img(image, target_size=(w, h))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = image.astype('float64')

preds = model.predict(image)
i = np.argmax(preds[0])


if args['layer'] == 'None':
	cam = GradCAM(model, i)
else:
    cam = GradCAM(model, i, args['layer'])
    
heatmap = cam.compute_heatmap(image)

# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

# draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (140, 40), (0, 0, 0), -1)
cv2.putText(output, "GradCAM", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.6, (255, 255, 255), 2)

# display the original image and resulting heatmap and output image
# to our screen
output = np.hstack([orig, heatmap, output])
output = imutils.resize(output, height=400)
cv2.imshow("Output", output)
cv2.waitKey(0)


output_path = args["output"]
cv2.imwrite(output_path, output)
print(f"Output image saved to: {output_path}")
