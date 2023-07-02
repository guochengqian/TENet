#!/usr/bin/env bash
# shopt -s extglob
conda activate ispnet

pipeline=$1
scale=$2
block=$3
n_blocks=$4
pretrain_dataset=$5 # pixelshift200, div2k
test_dataset=$6  # pixelshift200, urban100

pretrain=${PWD}/pretrain/${pretrain_dataset}/pipeline

# for all sequential solutions
#dn-dm-sr, dn-sr-dm, dm-dn-sr, dm-sr-dn, sr-dm-dn, sr-dn-dm, dn-dm+sr, dn+dm-sr, dn+sr-dm
python cmd_test_pipe_pixelshift.py --pipeline ${pipeline} --pretrain ${pretrain} --pretrain_dataset ${pretrain_dataset} --test_dataset ${test_dataset} --scale ${scale} --block ${block} --n_blocks ${n_blocks}

