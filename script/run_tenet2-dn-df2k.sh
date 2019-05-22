#!/usr/bin/env bash
python -u train.py\
	--train_list datasets/train_df2k.txt --valid_list datasets/valid_df2k.txt\
    --model tenet2 --bias --scale 2  --get2label\
    --denoise --max_noise 0.078 --min_noise 0.00\

# if run out of memory, lower batch_size down

