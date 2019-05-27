#!/usr/bin/env bash

python -u test.py --pretrained_model checkpoints/tenet2-dn-df2k.path --model tenet2 --scale 2 --bias --denoise --sigma 0 \
--test_path /data/TENet/TENet_test_data/TENet_real_test/sigma0/input/ --save_path output --crop_scale 8 --postname df2k


# if Run out of CUDA memory, just set --crop_scale 4 (or higher)
# remember to change sigma according to the noise level of the current image
# remember to change sigma according to the noise level of the current image