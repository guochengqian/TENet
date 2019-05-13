srun -p Superresolution  --gres=gpu:1 --job-name=denoraw python -u train_1gt.py\
	--train_list datasets/train_df2k.txt --valid_list datasets/valid_df2k.txt --lr 0.0001\
	--batch_size 16 --patch_size 64 --downsampler avg \
    --model denoraw --block_type rrdb --bias --scale 2\
    --denoise --max_noise 0.078 --min_noise 0.00\
