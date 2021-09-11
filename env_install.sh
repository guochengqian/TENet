#!/usr/bin/env bash
# make sure command is : source env_install.sh

# uncomment to install anaconda3.
#cd ~/
#wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
#bash Anaconda3-2019.07-Linux-x86_64.sh

# uncommet if using cluster
# module purge
# module load gcc
# module load cuda/10.1.243

# make sure your anaconda3 is added to bashrc (normally add to bashrc path automatically)
conda create -n tenet # conda create env
conda activate tenet  # activate

# step1: conda install and pip install
conda install pytorch=1.7 torchvision cudatoolkit=10.1 -c pytorch -y

# step2: install useful modules

pip install tqdm tensorboard opencv-python scipy scikit-image rawpy h5py wandb easydict


# Step3: install bicubic pytorch
git submodule add https://github.com/thstkdgus35/bicubic_pytorch
git submodule update --init --recursive

