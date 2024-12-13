#!/bin/bash
NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME=test_Inception_normal_32batch_normalization_29
EXP_DIR=/netscratch/mudraje/super_resolution_remote_sensing/logs/
export LOGLEVEL=INFO
#export CUDA_VISIBLE_DEVICES=1
#export MODEL_NAME="/netscratch/mudraje/super_resolution_remote_sensing/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"
# --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.07-py3.sqsh \
# --task-prolog="pwd/install.sh" \
srun -K\
    --job-name="${JOB_NAME}" \
    --partition="RTXA6000" \
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
    python -u /netscratch/mudraje/super_resolution_remote_sensing/bigearthnet-models-tf/eval.py /netscratch/mudraje/super_resolution_remote_sensing/bigearthnet-models-tf/configs/normal.json \
    