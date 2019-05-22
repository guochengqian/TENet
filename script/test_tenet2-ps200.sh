#!/usr/bin/env bash
python -u test.py --pretrained_model checkpoints/tennet2-dn-ps200.path --model tenet2 --scale 2 --bias \
--denoise --sigma 5 --test_path datasets/TENet_test_data/TENet_real_test/sigma5/input --save_path output

# if Run out of CUDA memory, just add     --crop_scale 2 (or higher)
# run ./script/test_tenet2-pixelshift200.sh