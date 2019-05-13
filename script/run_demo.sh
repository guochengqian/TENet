srun -p Superresolution  --gres=gpu:1 --job-name=demo python -u train_1gt.py\
	--train_list datasets/train_df2k.txt --valid_list datasets/valid_df2k.txt --lr 0.0001\
	--batch_size 16 --patch_size 64 --downsampler avg \
    --model demo --block_type rrdb --bias --scale 2\




