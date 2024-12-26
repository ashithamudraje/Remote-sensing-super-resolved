
# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
CAM visualization
"""

import argparse
import math
from io import BytesIO

import matplotlib.pyplot as plt
import requests
import tensorflow as tf
import torch
from PIL import Image
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
import torch.nn.functional as F
from torchcam import methods
from torchcam.utils import overlay_mask


def main(args):
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(args.device)

    # Load the PyTorch model architecture
    model = models.__dict__[args.arch](pretrained=False).to(device=device)
    #model.fc = torch.nn.Linear(model.fc.in_features, 19)
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_dir, map_location=device)
    
    # If the checkpoint is saved as a `state_dict`
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Freeze the model (no training, just inference)
    for p in model.parameters():
        p.requires_grad_(False)

    # Load and preprocess the image
    img_path = BytesIO(requests.get(args.img, timeout=5).content) if args.img.startswith("http") else args.img
    pil_img = Image.open(img_path, mode="r").convert("RGB")
    
    # Preprocess image for the model (normalize, resize, etc.)
    img_tensor = normalize(
        to_tensor(resize(pil_img, (224, 224))),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ).to(device=device)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    img_tensor.requires_grad_(True)

    # Get the CAM methods specified or use default
    if isinstance(args.method, str):
        cam_methods = args.method.split(",")
    else:
        cam_methods = [
            "CAM",
            #"GradCAM",
        ]

    # Hook the target layer and apply CAM method
    cam_extractors = [
        methods.__dict__[name](model, target_layer=args.target, enable_hooks=True) for name in cam_methods
    ]

    # Forward pass through the model before extracting CAM
    #with torch.no_grad():
    scores = model(img_tensor)
    print('score', scores)
    probs = torch.sigmoid(scores)
    print('probs', probs.shape)
    # Print all class probabilities
    # for idx, prob in enumerate(probs[0]):
    #     print(f"Class index: {idx}, Probability: {prob.item():.4f}")

    # Use the provided class index or the top predicted class
    class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx
    #print(f"Class index used for CAM: {class_idx}")
    class_prob = probs[0][class_idx].item()
    print(f"Class index: {class_idx}, Probability: {class_prob:.4f}")


    # Loop over all CAM methods and visualize
    num_cols = math.ceil((len(cam_extractors) + 1) / args.rows)
    _, axes = plt.subplots(args.rows, num_cols, figsize=(6, 4))
    
    # Display the input image
    ax = axes[0][0] if args.rows > 1 else axes[0] if num_cols > 1 else axes
    ax.imshow(pil_img)
    ax.set_title("Input", size=8)

    for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors):
        #extractor.enable_hooks()
        # Use the hooked data to compute activation map
        activation_map = extractor(class_idx, scores)[0].squeeze(0).cpu()

        # Clean up hooks after extracting activation maps
        # extractor.disable_hooks()
        # extractor.remove_hooks()
        
        # Convert activation map to heatmap and overlay on input image
        heatmap = to_pil_image(activation_map, mode="F")
        result = overlay_mask(pil_img, heatmap, alpha=args.alpha)

        ax = axes[idx // num_cols][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes
        ax.imshow(result)
        ax.set_title(extractor.__class__.__name__, size=8)

    # Clear axes for clean plotting
    if num_cols > 1:
        for _axes in axes:
            if args.rows > 1:
                for ax in _axes:
                    ax.axis("off")
            else:
                _axes.axis("off")
    else:
        axes.axis("off")

    plt.tight_layout()

    if args.savefig:
        plt.savefig(args.savefig, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    
    plt.show(block=not args.noblock)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Saliency Map comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch", type=str, default="resnet18", help="Name of the architecture")
    parser.add_argument(
        "--img",
        type=str,
        default="https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg",
        help="The image to extract CAM from",
    )
    parser.add_argument("--class-idx", type=int, default=None, help="Index of the class to inspect (default: top predicted)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Default device to perform computation on",
    )
    parser.add_argument("--savefig", type=str, default=None, help="Path to save figure")
    parser.add_argument("--method", type=str, default=None, help="CAM method to use")
    parser.add_argument("--target", type=str, default=None, help="The target layer for CAM")
    parser.add_argument("--alpha", type=float, default=0.5, help="Transparency of the heatmap overlay")
    parser.add_argument("--rows", type=int, default=1, help="Number of rows for the layout")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument(
        "--noblock",
        dest="noblock",
        help="Disables blocking visualization",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
