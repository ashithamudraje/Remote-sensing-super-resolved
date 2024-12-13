import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import os.path as osp
import sys
sys.path.append('/netscratch/mudraje/super_resolution_remote_sensing/BasicSR')
from basicsr.utils import img2tensor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append('/netscratch/mudraje/super_resolution_remote_sensing/HAT')
from hat.archs.hat_arch import HATArch

WINDOW_SIZE = 16
BATCH_SIZE = 32  # Adjust based on GPU memory

# Define the custom Dataset
class ImageDataset(Dataset):
    def __init__(self, image_dir, window_size):
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(".jpg")
        ]
        self.window_size = window_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = resize_image_to_fit_window(img, self.window_size)
        #(f"Type of img: {type(img)}")
        return img, img_path

def resize_image_to_fit_window(img, window_size):
    h, w, _ = img.shape
    new_h = (h // window_size) * window_size
    new_w = (w // window_size) * window_size
    resized_img = cv2.resize(img, (new_w, new_h))
    return resized_img

# Adjusted HAT class to support multi-GPU
class HAT:
    def __init__(
        self,
        upscale=2,
        in_chans=3,
        img_size=(480, 640),
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    ):
        upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the HAT model
        self.sr_model = HATArch(
            img_size=img_size,
            upscale=upscale,
            in_chans=in_chans,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            img_range=img_range,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            resi_connection=resi_connection,
        ).to(self.device)

        # Load model weights
        ckpt_path = os.path.join(
            ROOT_DIR,
            "HAT",
            "weights",
            "HAT_SRx4_ImageNet-pretrain.pth",
        )
        loadnet = torch.load(ckpt_path, map_location=self.device)
        keyname = "params_ema" if "params_ema" in loadnet else "params"
        self.sr_model.load_state_dict(loadnet[keyname])
        self.sr_model.eval()

        # Support for multi-GPU
        if torch.cuda.device_count() > 1:
            self.sr_model = DataParallel(self.sr_model)

    @torch.no_grad()
    def __call__(self, img_batch):
        # Convert images to tensor and normalize to [0, 1]
        sr_img_batch=[]
        for img in img_batch:
            #print(f"Type of img: {type(img)}")
            if isinstance(img, torch.Tensor):
                # Convert tensor to numpy array for debugging
                img = img.cpu().numpy()

            img_tensor = (
                img2tensor(imgs=img / 255.0, bgr2rgb=True, float32=True).unsqueeze(0).to(self.device)
            )
            restored_img = self.sr_model(img_tensor)[0]
            restored_img = restored_img.permute(1, 2, 0).cpu().numpy()
            restored_img = (restored_img - restored_img.min()) / (
                restored_img.max() - restored_img.min()
            )
            restored_img = (restored_img * 255).astype(np.uint8)
            restored_img = Image.fromarray(restored_img)
            restored_img = np.array(restored_img)
            sr_img = cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR)
            sr_img_batch.append(sr_img)

        return sr_img_batch

# Main function to process the images in batches
def process_images_in_batches(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    dataset = ImageDataset(input_dir, WINDOW_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    hat = HAT(upscale=4)

    for img_batch, paths in dataloader:
        # print(f"Type of img_batch: {type(img_batch)}")
        # print(f"Type of img_batch[0]: {type(img_batch[0])}")
        sr_img_batch = hat(img_batch)
        for sr_img, path in zip(sr_img_batch, paths):
            save_path = os.path.join(output_dir, os.path.basename(path))
            cv2.imwrite(save_path, sr_img)
            print(f"Processed and saved: {save_path}")

    print("Processing completed.")

if __name__ == "__main__":
    input_dir = "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb/images"
    output_dir = "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/Hat_all_images"
    process_images_in_batches(input_dir, output_dir)
