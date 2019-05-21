#!/usr/bin/env bash
python -u test.py --pretrained_model checkpoints/tennet2-dn-pixelshit200.path --model tenet2 --scale 2 --bias --denoise --sigma 10 --test_path datasets/input --save_path output

# if Run out of CUDA memory, just add     --crop_scale 2 (or higher)
# run ./script/test_tenet2-pixelshift200.sh