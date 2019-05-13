srun -p Superresolution  --gres=gpu:1 --job-name=tri2dn python -u train_2gt.py \
	--train_list datasets/train_mat.txt --valid_list datasets/valid_mat.txt \
	--lr 0.0001 --batch_size 16 --patch_size 128\
    --model tri2 --block_type rrdb --sr_n_resblocks 6 --dm_n_resblocks 6 --scale 2 --bias --channels 64 \
    --denoise --max_noise 0.07 --min_noise 0.00 --get2label --downsampler bic --postname mat\
