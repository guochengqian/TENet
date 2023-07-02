#!/usr/bin/env bash
# shopt -s extglob

#conda activate ispnet

# for full joint methods
################# pipeline1: tenet-dn+sr-dm
scale=$1
pipeline=$2 # e2e-dn+sr-dm, name of this pipeline
in_type=$3
mid_type=$4
out_type=$5
model=$6
block=$7
n_blocks=$8
pretrain_dataset=$9 # pixelshift200, div2k
benchmark_dataset=${10}  # pixelshift200, urban100
noise_model=${11}

pretrain="pretrain/${pretrain_dataset}/pipeline"

pretrain=${PWD}/pretrain/${pretrain_dataset}/pipeline
save_dir=${pretrain}/result_${benchmark_dataset}_${noise_model}x${scale}/${pipeline}/result
rm -rf ${save_dir}
mkdir -p ${save_dir}

echo "save to ${save_dir}"
python test_benchmark.py --phase test --in_type ${in_type} --mid_type ${mid_type} --out_type ${out_type} --model ${model} --block ${block}  --n_blocks ${n_blocks} --scale ${scale} --pretrain ${pretrain} --save_dir ${save_dir}  --dataset ${pretrain_dataset} --test_dataset ${benchmark_dataset} --noise_model ${noise_model}