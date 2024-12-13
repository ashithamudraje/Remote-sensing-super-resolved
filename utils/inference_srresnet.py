import os
import cv2
import sys
import torch
import numpy as np
import os.path as osp
from PIL import Image
sys.path.append('/netscratch/mudraje/super_resolution_remote_sensing/BasicSR')
from basicsr.utils import img2tensor
from basicsr.archs.srresnet_arch import MSRResNet

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)


class SRResNet:

    def __init__(self, upscale=2, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16):

        self.upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------ load model for img enhancement -------------------
        self.sr_model = MSRResNet(
            upscale=self.upscale,
            num_in_ch=num_in_ch,
            num_out_ch=num_out_ch,
            num_feat=num_feat,
            num_block=num_block,
        ).to(self.device)

        ckpt_path = os.path.join(
            ROOT_DIR,
            "srresnet",
            "weights",
            f"SRResNet_{str(self.upscale)}x.pth",
        )
        loadnet = torch.load(ckpt_path, map_location=self.device)
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"

        self.sr_model.load_state_dict(loadnet[keyname])
        self.sr_model.eval()

    @torch.no_grad()
    def __call__(self, img):
        img_tensor = (
            img2tensor(imgs=img / 255.0, bgr2rgb=True, float32=True)
            .unsqueeze(0)
            .to(self.device)
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

        return sr_img


if __name__ == "__main__":

    srresnet = SRResNet(upscale=4)

    # img = cv2.imread(f"{ROOT_DIR}/data/EyeDentify/Wo_SR/original/1/1/frame_01.png")
    input_dir = "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb/images"
    output_dir = "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/SRResNet_all_imgs"

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):  # Add more extensions if needed
        # Read the image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # Check if the image was read successfully
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            # Apply the Real-ESRGAN model
            sr_img = srresnet(img=img)

            # Save the super-resolved image with the original filename
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, sr_img)

            print(f"Processed and saved: {save_path}")

    print("Processing completed.")