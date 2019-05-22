#!/usr/bin/env bash
python -u test.py --pretrained_model checkpoints/tenet2-dn-df2k.path --model tenet2 --scale 2 --bias --denoise --sigma 10 \
--test_path dataset/TENet_test_data/TENet_sim_test/noisy_input/sigma10/mcm --save_path output

# if Run out of CUDA memory, just set --crop_scale 2 (or higher)
# remember to change sigma according to the noise level of the current image
