#!/usr/bin/env bash
# make sure command is : source env_install.sh

conda activate ispnet  # activate

in_type=$1
mid_type=$2
out_type=$3
model=$4
pretrain=$5
PY_ARGS=${@:6}
CUDA_VISIBLE_DEVICES=1 python test_benchmark.py --phase test --in_type ${in_type} --mid_type ${mid_type} --out_type ${out_type} --model ${model} --pretrain ${pretrain} ${PY_ARGS}


