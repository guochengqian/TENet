#!/usr/bin/env bash
python -u test.py --pretrained_model checkpoints/tenet2-dn-ps200.path  \
--scale 2 --bias --model tenet2 --denoise --sigma 0 --postname ps200 --crop_scale 8 --show_info \
--test_path /data/image/TENet/TENet_test_data/TENet_real_test/sigma0/input --save_path output


