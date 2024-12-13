import os
import cv2
import sys
import torch
import os.path as osp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
sys.path.append(root_path)
from inference_sr_utils import RealEsrUpsamplerZoo


class RealEsr:

    def __init__(
        self,
        upscale=2,
        bg_upsampler_name="realesrgan",
        prefered_net_in_upsampler="RRDBNet",
    ):

        self.upscale = int(upscale)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------ set up background upsampler ------------------------
        self.upsampler_zoo = RealEsrUpsamplerZoo(
            upscale=self.upscale,
            bg_upsampler_name=bg_upsampler_name,
            prefered_net_in_upsampler=prefered_net_in_upsampler,
        )
        self.bg_upsampler = self.upsampler_zoo.bg_upsampler

    def __call__(self, img):
        # ---------------- restore/enhance image using the selected RealESR model ----------------
        sr_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]

        return sr_img


if __name__ == "__main__":

    realesr = RealEsr(
        upscale=4, bg_upsampler_name="realesrgan", prefered_net_in_upsampler="RRDBNet"
    )

    # img = cv2.imread(f"{ROOT_DIR}/data/EyeDentify/Wo_SR/original/1/1/frame_01.png")
    # img = cv2.imread(
    #     f"{ROOT_DIR}/data/EyeDentify/Wo_SR/eyes/left_eyes/23/1/frame_01.png"
    # )
    # img = cv2.imread(
    #     "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/train_images"
    # )
    # sr_img = realesr(img=img)

    # saving_dir = "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/RealESRgan_imgs"
    # os.makedirs(saving_dir, exist_ok=True)
    # cv2.imwrite(f"{saving_dir}/sr_img_realesrgan_1_15.png", sr_img)
    input_dir = "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb/images"
    output_dir = "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/RealESRgan_all_imgs"

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
            sr_img = realesr(img=img)

            # Save the super-resolved image with the original filename
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, sr_img)

            print(f"Processed and saved: {save_path}")

    print("Processing completed.")