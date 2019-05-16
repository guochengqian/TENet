python -u train_2gt.py\
	--train_list datasets/train_df2k.txt --valid_list datasets/valid_df2k.txt --lr 0.0001\
    --model tenet2 --block_type rrdb --bias --scale 2\
    --denoise --max_noise 0.078 --min_noise 0.00\

