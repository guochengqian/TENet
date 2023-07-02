#!/usr/bin/env bash
# make sure command is : source env_install.sh

# uncomment to install anaconda3.
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
# bash Miniconda3-latest-Linux-x86_64.sh  
# source ~/.bashrc

export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1
module purge
module load cuda/11.1.1
module load gcc
# make sure local cuda version is 11.1

# install PyTorch
conda deactivate
conda env remove --name tenet 
conda create -n tenet -y python=3.7 numpy=1.20 numba
conda activate tenet

# #NOTE: 'nvidia' channel is required for cudatoolkit 11.1 with pytorch version 1.10.x
conda install -y pytorch=1.10.1 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
# install bicubic pytorch
git submodule update --init --recursive

pip install -r requirements.txt