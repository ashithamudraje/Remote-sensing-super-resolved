#!/bin/bash
# put your install commands here (remove lines you don't need):
# make sure you run chmod a+x install.sh in cluster to give permission to this batch file
# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  # put your install commands here (remove lines you don't need):
  apt update
  apt clean
#pip install -r ./requirements.txt

  # If encountered issue with opencv lib, use one of the following fixes:

  # Fix-1.
  pip install numpy
  pip install pandas
  pip install scikit-image
  pip install torch==1.13.1
  pip install pillow
  # pip install huggingface_hub
  pip install accelerate
  pip install einops
  pip install mlflow
  #pip install keras
  pip install tensorflow==2.7 # pip install protobuf==3.20.*
  pip install transformers==4.25.0
  pip install xformers    
  pip install torchvision
  pip install datasets
  pip install tf-slim
  pip install opencv-python-headless==4.5.5.64
  pip install protobuf==3.20.*
  pip install torchcam
  pip install imutils
  pip install diffusers==0.21.0
  pip install timm
  pip install loralib
  pip install fairscale
  pip install pydantic==1.10.11
  pip install gradio==3.24.0
  pip install pytorch_lightning
#   pip install onnx2torch

  # OR
  
  # Fix-2.
  # pip install opencv-fixer==0.2.5
  # python3 -c "from opencv_fixer import AutoFix; AutoFix()"
  # pip3 install opencv-python --upgrade

  apt-get clean
  
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi