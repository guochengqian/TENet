#!/usr/bin/env bash

# RRG
sbatch -J rrg slurm_train_1job.sh noisy_lr_raw raw linrgb tenet --block rrg --batch_per_gpu 32 --n_blocks 1
