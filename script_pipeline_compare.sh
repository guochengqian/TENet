#!/usr/bin/env bash
# shopt -s extglob

conda activate ispnet

pipeline=$1
scale=$2
block_type=$3
n_blocks=$4
pretrain_dataset=$5 # pixelshift200, div2k
benchmark_dataset=$6  # pixelshift200, urban100

benchmark_path=data/benchmark/${benchmark_dataset}/${benchmark_dataset}_noisy_lr_raw_rgb_x${scale}.pt
pretrain_dir=${PWD}/pretrain/${pretrain_dataset}/pipeline

# for all sequential solutions
#dn-dm-sr, dn-sr-dm, dm-dn-sr, dm-sr-dn, sr-dm-dn, sr-dn-dm, dn-dm+sr, dn+dm-sr, dn+sr-dm
python cmd_test_pipe_pixelshift.py --pipeline ${pipeline} --pretrain_dir ${pretrain_dir} --test_data ${benchmark_path} --scale ${scale} --block_type ${block_type} --n_blocks ${n_blocks}

