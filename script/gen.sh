#!/bin/bash
NOW=$( date '+%F-%H-%M-%S' )
JOB_NAME=split_gen
EXP_DIR=/netscratch/mudraje/super_resolution_remote_sensing/logs/
export LOGLEVEL=INFO
#export TF_TRT_USE_NATIVE_SEGMENT_EXECUTION=1
#export CUDA_VISIBLE_DEVICES=1
#export MODEL_NAME="/netscratch/mudraje/super_resolution_remote_sensing/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64"
# --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.07-py3.sqsh \
# --task-prolog="pwd/install.sh" \
srun -K\
    --job-name="${JOB_NAME}" \
    --partition="A100-40GB" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=0 \
    --cpus-per-task=4 \
    --mem=70G \
    --container-image=/netscratch/mudraje/super_resolution_remote_sensing/resolution_dependencies.sqsh \
    --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds,/ds-sds:/ds-sds,"`pwd`":"`pwd`" \
    --container-workdir="`pwd`" \
    --time 72:00:00 \
    --output="${EXP_DIR}/${NOW}-${JOB_NAME}.log" \
    python -u /netscratch/mudraje/super_resolution_remote_sensing/BigEarthNet-S2_19-classes_models/prep_splits_19_classes.py \
    -r "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/Seesr_all_imgs/sample00" \
    -o "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/Seesr_all_splits" \
    -label "/ds/images/BigEarthNet/BigEarthNet-S2/metadata/BigEarthNet-v1.0" \
    -n "/netscratch/mudraje/super_resolution_remote_sensing/BigEarthNet-S2_19-classes_models/splits/train.csv" \
       "/netscratch/mudraje/super_resolution_remote_sensing/BigEarthNet-S2_19-classes_models/splits/val.csv" \
       "/netscratch/mudraje/super_resolution_remote_sensing/BigEarthNet-S2_19-classes_models/splits/test.csv" \
    -l "tensorflow" \
