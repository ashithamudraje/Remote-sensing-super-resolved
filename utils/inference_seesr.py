import os
import sys
sys.path.append(os.getcwd())
import cv2
import glob
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
sys.path.append('/netscratch/mudraje/super_resolution_remote_sensing/SeeSR/')
from pipelines.pipeline_seesr import StableDiffusionControlNetPipeline
from utils.misc import load_dreambooth_lora
from utils.wavelet_color_fix import wavelet_color_fix, adain_color_fix

from ram.models.ram_lora import ram
from ram import inference_ram as inference
from ram import get_transform

from typing import Mapping, Any
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

logger = get_logger(__name__, log_level="INFO")

tensor_transforms = transforms.Compose([
    transforms.ToTensor(),
])

ram_transforms = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_state_dict_diffbirSwinIR(model: nn.Module, state_dict: Mapping[str, Any], strict: bool=False) -> None:
    state_dict = state_dict.get("state_dict", state_dict)
    
    is_model_key_starts_with_module = list(model.state_dict().keys())[0].startswith("module.")
    is_state_dict_key_starts_with_module = list(state_dict.keys())[0].startswith("module.")
    
    if is_model_key_starts_with_module and not is_state_dict_key_starts_with_module:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    if not is_model_key_starts_with_module and is_state_dict_key_starts_with_module:
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    
    model.load_state_dict(state_dict, strict=strict)

def load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    from models.controlnet import ControlNetModel
    from models.unet_2d_condition import UNet2DConditionModel

    # Load scheduler, tokenizer, and models
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.seesr_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.seesr_model_path, subfolder="controlnet")
    
    # Freeze models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly.")

    # Initialize pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

    # Set weight dtype based on precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to device and cast to appropriate dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    return validation_pipeline

def load_tag_model(args, device='cuda'):
    model = ram(pretrained='/netscratch/mudraje/super_resolution_remote_sensing/ram_swin_large_14m.pth',
                pretrained_condition=args.ram_ft_path,
                image_size=384,
                vit='swin_l')
    model.eval()
    model.to(device)
    return model

def get_validation_prompt(args, image, model, device='cuda'):
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    res = inference(lq, model)
    ram_encoder_hidden_states = model.generate_image_embeds(lq)
    validation_prompt = f"{res[0]}, {args.prompt},"
    return validation_prompt, ram_encoder_hidden_states

def process_image(image_path, pipeline, model, args, accelerator, generator):
    validation_image = Image.open(image_path).convert("RGB")
    validation_prompt, ram_encoder_hidden_states = get_validation_prompt(args, validation_image, model)
    validation_prompt += args.added_prompt
    negative_prompt = args.negative_prompt
    print(f'{validation_prompt}')

    ori_width, ori_height = validation_image.size
    resize_flag = False
    rscale = args.upscale
    if ori_width < args.process_size // rscale or ori_height < args.process_size // rscale:
        scale = (args.process_size // rscale) / min(ori_width, ori_height)
        validation_image = validation_image.resize((int(scale * ori_width), int(scale * ori_height)))
        resize_flag = True

    validation_image = validation_image.resize((validation_image.size[0] * rscale, validation_image.size[1] * rscale))
    validation_image = validation_image.resize((validation_image.size[0] // 8 * 8, validation_image.size[1] // 8 * 8))
    width, height = validation_image.size

    with torch.autocast("cuda"):
        generated_image = pipeline(
            validation_prompt, validation_image, num_inference_steps=args.num_inference_steps, generator=generator,
            height=height, width=width, guidance_scale=args.guidance_scale, negative_prompt=negative_prompt,
            conditioning_scale=args.conditioning_scale, start_point=args.start_point,
            ram_encoder_hidden_states=ram_encoder_hidden_states, latent_tiled_size=args.latent_tiled_size,
            latent_tiled_overlap=args.latent_tiled_overlap, args=args
        ).images[0]
    
        if args.align_method == 'nofix':
            generated_image = generated_image
        else:
            if args.align_method == 'wavelet':
                generated_image = wavelet_color_fix(generated_image, validation_image)
            elif args.align_method == 'adain':
                generated_image = adain_color_fix(generated_image, validation_image)

    if resize_flag:
        generated_image = generated_image.resize((ori_width * rscale, ori_height * rscale))
    
    name, ext = os.path.splitext(os.path.basename(image_path))
    save_path = f'{args.output_dir}/sample00/{name}.jpg'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    generated_image.save(save_path)

def main(args, enable_xformers_memory_efficient_attention=True):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    
    if args.seed is not None:
        set_seed(args.seed)

    # Load models
    pipeline = load_seesr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)
    model = load_tag_model(args, accelerator.device)
    
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    if os.path.isdir(args.image_path):
        image_names = sorted(glob.glob(f'{args.image_path}/*.*'))
    else:
        image_names = [args.image_path]

    # Process images within the specified index range
    start_idx = max(0, args.start_idx)
    end_idx = min(len(image_names), args.end_idx)
    image_names_to_process = image_names[start_idx:end_idx]

    for image_idx, image_name in enumerate(image_names_to_process):
        print(f'Processing {image_idx + start_idx + 1}/{end_idx} image: {image_name}')
        process_image(image_name, pipeline, model, args, accelerator, generator)
        torch.cuda.empty_cache()  # Free up memory after each image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seesr_model_path", type=str, default=None)
    parser.add_argument("--ram_ft_path", type=str, default=None)
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="dotted, noise, blur, lowres, smooth")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=224)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024)
    parser.add_argument("--latent_tiled_size", type=int, default=96)
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--start_point", type=str, choices=['lr', 'noise'], default='lr')
    parser.add_argument("--save_prompts", action='store_true')
    parser.add_argument("--start_idx", type=int, default=0)  # Start index for image processing
    parser.add_argument("--end_idx", type=int, default=25000)  # End index for image processing
    args = parser.parse_args()
    main(args)
