#!/bin/bash
NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME=image_resolve
EXP_DIR=/netscratch/mudraje/super_resolution_remote_sensing/logs/
export LOGLEVEL=INFO
export CUDA_VISIBLE_DEVICES=1
#export MODEL_NAME="/netscratch/mudraje/super_resolution_remote_sensing/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"
# --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.07-py3.sqsh \
# --task-prolog="pwd/install.sh" \
srun -K\
    --job-name="${JOB_NAME}" \
    --partition="V100-32GB" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=1 \
    --cpus-per-task=4 \
    --mem=70G \
    --container-image=/netscratch/mudraje/super_resolution_remote_sensing/resolution_dependencies.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time 72:00:00 \
    --output="${EXP_DIR}/${NOW}-${JOB_NAME}.log" \
    python -u /netscratch/mudraje/super_resolution_remote_sensing/utils/cam_normal.py --arch resnet152 --savefig /netscratch/mudraje/super_resolution_remote_sensing/res152_srresnet_tf_with_resolved/results/cam_srresnet.jpg --class-idx 4 \
            --img /ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/SRResNet_all_imgs/S2A_MSIL2A_20180507T093041_18_3.jpg --checkpoint_dir /netscratch/mudraje/super_resolution_remote_sensing/converted_checkpoints/converted_srresnet_resnet152.pth \
    # python -u /netscratch/mudraje/super_resolution_remote_sensing/utils/cam_normal.py --arch resnet152 --savefig /netscratch/mudraje/super_resolution_remote_sensing/res512_normal/results/cam_normal_45_69.jpg --class-idx 4 \
    #         --img /ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb/images/S2A_MSIL2A_20170613T101031_45_69.jpg --checkpoint_dir /netscratch/mudraje/super_resolution_remote_sensing/converted_checkpoints/converted_normal_resnet152.pth \
    # python -u /netscratch/mudraje/super_resolution_remote_sensing/utils/cam_normal.py --arch resnet152 --savefig /netscratch/mudraje/super_resolution_remote_sensing/res152_srresnet_tf_with_resolved/results/cam_srresnet.jpg --class-idx 5 \
    #         --img /ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/SRResNet_all_imgs/S2A_MSIL2A_20180507T093041_18_3.jpg --checkpoint_dir /netscratch/mudraje/super_resolution_remote_sensing/converted_checkpoints/converted_resnet152.pth \
    # python -u /netscratch/mudraje/super_resolution_remote_sensing/utils/cam_normal.py --arch resnet152 --savefig /netscratch/mudraje/super_resolution_remote_sensing/res512_hat/results/gradcam_output_hat1.jpg --class-idx 5 \
    #         --img /ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/Hat_all_images/S2A_MSIL2A_20180507T093041_18_3.jpg --checkpoint_dir /netscratch/mudraje/super_resolution_remote_sensing/converted_checkpoints/converted_hat-resnet152.pth \
