#!/usr/bin/env bash
python -u test.py --pretrained_model checkpoints/tenet2-dn-ps200.path --model tenet2 --scale 2 --bias  \
--denoise --sigma 0 --test_path /data/TENet/TENet_test_data/TENet_real_test/sigma0/input --save_path output --postname ps200 --crop_scale 8

# if Run out of CUDA memory, just add     --crop_scale 2 (or higher)
# run ./script/test_tenet2-pixelshift200.sh

