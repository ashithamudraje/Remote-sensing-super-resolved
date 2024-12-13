#!/bin/bash
NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME=inference_seesr
EXP_DIR=/netscratch/mudraje/super_resolution_remote_sensing/logs/
export LOGLEVEL=INFO
#export CUDA_VISIBLE_DEVICES=1
#export MODEL_NAME="/netscratch/mudraje/super_resolution_remote_sensing/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"
# --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.07-py3.sqsh \
# --task-prolog="pwd/install.sh" \
srun -K\
    --job-name="${JOB_NAME}" \
    --partition="batch" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=1 \
    --cpus-per-task=4 \
    --mem=90G \
    --container-image=/netscratch/mudraje/super_resolution_remote_sensing/resolution_dependencies.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time 72:00:00 \
    --output="${EXP_DIR}/${NOW}-${JOB_NAME}.log" \
    python -u /netscratch/mudraje/super_resolution_remote_sensing/utils/inference_seesr.py \
    --pretrained_model_path /netscratch/mudraje/super_resolution_remote_sensing/sd2 \
    --prompt '' \
    --seesr_model_path /netscratch/mudraje/super_resolution_remote_sensing/seesr \
    --ram_ft_path /netscratch/mudraje/super_resolution_remote_sensing/DAPE.pth \
    --image_path /ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb/images \
    --output_dir /ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/Seesr_all_imgs \
    --start_point lr \
    --num_inference_steps 20 \
    --guidance_scale 5.5 \
    --process_size 512 \
    --start_idx 500000 \
    --end_idx 519284