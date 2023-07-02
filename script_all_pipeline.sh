#!/usr/bin/env bash
# shopt -s extglob

# Test on pixelshift
## sequential
### SR 2
# bash script_pipeline_compare.sh dm-dn-sr 2 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dm-sr-dn 2 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh sr-dm-dn 2 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh sr-dn-dm 2 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dn-dm-sr 2 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dn-sr-dm 2 rrdb 6 pixelshift pixelshift

# evalue performance
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx2/dm-dn-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx2/dm-sr-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx2/sr-dm-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx2/sr-dn-dm/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx2/dn-dm-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx2/dn-sr-dm/result --pred_pattern output

## partial joint
# bash script_pipeline_compare.sh dn-dm+sr 2 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dn+dm-sr 2 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dn+sr-dm 2 rrdb 6 pixelshift pixelshift
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_pixelshift_gpx2/dn-dm+sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx2/dn+dm-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx2/dn+sr-dm/result --pred_pattern output


## joint
# bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 12 pixelshift pixelshift gp
# bash script_e2e_compare.sh 2 e2e-dn-dm+sr noisy_lr_raw lr_raw linrgb tenet rrdb 12 pixelshift pixelshift gp
# bash script_e2e_compare.sh 2 e2e-dn+dm-sr noisy_lr_raw lr_linrgb linrgb tenet rrdb 12 pixelshift pixelshift gp
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 pixelshift pixelshift gp
# bash script_e2e_compare.sh 2 e2e-dn-sr-dm noisy_lr_raw lr_raw,raw linrgb tenetv2 rrdb 12 pixelshift pixelshift


### SR 4
# bash script_pipeline_compare.sh dm-dn-sr 4 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dm-sr-dn 4 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh sr-dm-dn 4 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh sr-dn-dm 4 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dn-dm-sr 4 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dn-sr-dm 4 rrdb 6 pixelshift pixelshift

# evalue performance
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx4/dm-dn-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx4/dm-sr-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx4/sr-dm-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx4/sr-dn-dm/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx4/dn-dm-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_gpx4/dn-sr-dm/result --pred_pattern output



# bash script_pipeline_compare.sh dn-dm+sr 4 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dn+dm-sr 4 rrdb 6 pixelshift pixelshift
# bash script_pipeline_compare.sh dn+sr-dm 4 rrdb 6 pixelshift pixelshift

# python evaluate_metrics.py --test_dataset pixelshift --pred_dir pretrain/pixelshift/pipeline/results_pixelshift_gpx4/dn-dm+sr/result --pred_pattern output

# bash script_e2e_compare.sh 4 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 12 pixelshift pixelshift gp
# bash script_e2e_compare.sh 4 e2e-dn-dm+sr noisy_lr_raw lr_raw linrgb tenet rrdb 12 pixelshift pixelshift gp
# bash script_e2e_compare.sh 4 e2e-dn+dm-sr noisy_lr_raw lr_linrgb linrgb tenet rrdb 12 pixelshift pixelshift gp
# bash script_e2e_compare.sh 4 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 pixelshift pixelshift gp


## E2E using SOTA models.
### jdsr
bash script_e2e_compare.sh 2 jdsr-dn+dm+sr noisy_lr_raw None linrgb jdsr rrdb 24 pixelshift pixelshift gp
bash script_e2e_compare.sh 2 jdsr-dn+sr-dm noisy_lr_raw raw linrgb jdsr rrdb 24 pixelshift pixelshift gp

### jdndmsr
#bash script_e2e_compare.sh 2 jdndmsr-dn+dm+sr noisy_lr_raw None linrgb jdndmsr rrdb 4 pixelshift pixelshift gp
# bash script_e2e_compare.sh 2 jdndmsr-dn+sr-dm noisy_lr_raw raw linrgb jdndmsr rrdb 2 pixelshift pixelshift gp
#bash script_e2e_compare.sh 2 jdndmsr-dn+sr-dm noisy_lr_raw raw linrgb jdndmsr rrdb 4 pixelshift pixelshift gp

## E2E using other modules.
### Res
#bash script_e2e_compare.sh 2 resnet-dn+dm+sr noisy_lr_raw None linrgb tenet res 6  pixelshift pixelshift
#bash script_e2e_compare.sh 2 resnet-dn+sr-dm noisy_lr_raw raw linrgb tenet res 6  pixelshift pixelshift
#
### NLSA
# bash script_e2e_compare.sh 2 nlsa-dn+dm+sr noisy_lr_raw None linrgb tenet nlsa 12 pixelshift pixelshift
# bash script_e2e_compare.sh 2 nlsa-dn+sr-dm noisy_lr_raw raw linrgb tenet nlsa 12 pixelshift pixelshift


# Ablate dataset
## sequential
# bash script_pipeline_compare.sh dm-dn-sr 2 rrdb 6 div2k urban100
# bash script_pipeline_compare.sh dm-sr-dn 2 rrdb 6 div2k urban100
# bash script_pipeline_compare.sh sr-dm-dn 2 rrdb 6 div2k urban100
# bash script_pipeline_compare.sh sr-dn-dm 2 rrdb 6 div2k urban100
# bash script_pipeline_compare.sh dn-dm-sr 2 rrdb 6 div2k urban100
# bash script_pipeline_compare.sh dn-sr-dm 2 rrdb 6 div2k urban100
# ### partial joint
# bash script_pipeline_compare.sh dn-dm+sr 2 rrdb 6 div2k urban100
# bash script_pipeline_compare.sh dn+dm-sr 2 rrdb 6 div2k urban100
# bash script_pipeline_compare.sh dn+sr-dm 2 rrdb 6 div2k urban100

# python evaluate_metrics.py --test_dataset urban100 --pred_dir pretrain/div2k/pipeline/results_urban100_gpx2/dm-dn-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset urban100 --pred_dir pretrain/div2k/pipeline/results_urban100_gpx2/dm-sr-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset urban100 --pred_dir pretrain/div2k/pipeline/results_urban100_gpx2/sr-dm-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset urban100 --pred_dir pretrain/div2k/pipeline/results_urban100_gpx2/sr-dn-dm/result --pred_pattern output
# python evaluate_metrics.py --test_dataset urban100 --pred_dir pretrain/div2k/pipeline/results_urban100_gpx2/dn-dm-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset urban100 --pred_dir pretrain/div2k/pipeline/results_urban100_gpx2/dn-sr-dm/result --pred_pattern output

# python evaluate_metrics.py --test_dataset urban100 --pred_dir pretrain/div2k/pipeline/results_urban100_gpx2/dn-dm+sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset urban100 --pred_dir pretrain/div2k/pipeline/results_urban100_gpx2/dn+dm-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset urban100 --pred_dir pretrain/div2k/pipeline/results_urban100_gpx2/dn+sr-dm/result --pred_pattern output

## joint
# bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 12 div2k urban100
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 div2k urban100

# # DIV2K
# ## sequential
# bash script_pipeline_compare.sh dm-dn-sr 2 rrdb 6 div2k div2k
# bash script_pipeline_compare.sh dm-sr-dn 2 rrdb 6 div2k div2k
# bash script_pipeline_compare.sh sr-dm-dn 2 rrdb 6 div2k div2k
# bash script_pipeline_compare.sh sr-dn-dm 2 rrdb 6 div2k div2k
# bash script_pipeline_compare.sh dn-dm-sr 2 rrdb 6 div2k div2k
# bash script_pipeline_compare.sh dn-sr-dm 2 rrdb 6 div2k div2k
# ### partial joint
# bash script_pipeline_compare.sh dn-dm+sr 2 rrdb 6 div2k div2k
# bash script_pipeline_compare.sh dn+dm-sr 2 rrdb 6 div2k div2k
# bash script_pipeline_compare.sh dn+sr-dm 2 rrdb 6 div2k div2k

# python evaluate_metrics.py --test_dataset div2k --pred_dir pretrain/div2k/pipeline/results_div2k_gpx2/dm-dn-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset div2k --pred_dir pretrain/div2k/pipeline/results_div2k_gpx2/dm-sr-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset div2k --pred_dir pretrain/div2k/pipeline/results_div2k_gpx2/sr-dm-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset div2k --pred_dir pretrain/div2k/pipeline/results_div2k_gpx2/sr-dn-dm/result --pred_pattern output
# python evaluate_metrics.py --test_dataset div2k --pred_dir pretrain/div2k/pipeline/results_div2k_gpx2/dn-dm-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset div2k --pred_dir pretrain/div2k/pipeline/results_div2k_gpx2/dn-sr-dm/result --pred_pattern output

# python evaluate_metrics.py --test_dataset div2k --pred_dir pretrain/div2k/pipeline/results_div2k_gpx2/dn-dm+sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset div2k --pred_dir pretrain/div2k/pipeline/results_div2k_gpx2/dn+dm-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset div2k --pred_dir pretrain/div2k/pipeline/results_div2k_gpx2/dn+sr-dm/result --pred_pattern output

# # # joint
# bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 12 div2k div2k gp
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 div2k div2k gp
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 div2k div2k gp

# ## sequential
# bash script_pipeline_compare.sh dm-dn-sr 2 rrdb 6 div2k cbsd68 
# bash script_pipeline_compare.sh dm-sr-dn 2 rrdb 6 div2k cbsd68 
# bash script_pipeline_compare.sh sr-dm-dn 2 rrdb 6 div2k cbsd68 
# bash script_pipeline_compare.sh sr-dn-dm 2 rrdb 6 div2k cbsd68 
# bash script_pipeline_compare.sh dn-dm-sr 2 rrdb 6 div2k cbsd68 
# bash script_pipeline_compare.sh dn-sr-dm 2 rrdb 6 div2k cbsd68 
# ### partial 
# bash script_pipeline_compare.sh dn-dm+sr 2 rrdb 6 div2k cbsd68
# bash script_pipeline_compare.sh dn+dm-sr 2 rrdb 6 div2k cbsd68
# bash script_pipeline_compare.sh dn+sr-dm 2 rrdb 6 div2k cbsd68

# python evaluate_metrics.py --test_dataset cbsd68 --pred_dir pretrain/div2k/pipeline/results_cbsd68_gpx2/dm-dn-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset cbsd68 --pred_dir pretrain/div2k/pipeline/results_cbsd68_gpx2/dm-sr-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset cbsd68 --pred_dir pretrain/div2k/pipeline/results_cbsd68_gpx2/sr-dm-dn/result --pred_pattern output
# python evaluate_metrics.py --test_dataset cbsd68 --pred_dir pretrain/div2k/pipeline/results_cbsd68_gpx2/sr-dn-dm/result --pred_pattern output
# python evaluate_metrics.py --test_dataset cbsd68 --pred_dir pretrain/div2k/pipeline/results_cbsd68_gpx2/dn-dm-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset cbsd68 --pred_dir pretrain/div2k/pipeline/results_cbsd68_gpx2/dn-sr-dm/result --pred_pattern output

# python evaluate_metrics.py --test_dataset cbsd68 --pred_dir pretrain/div2k/pipeline/results_cbsd68_gpx2/dn-dm+sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset cbsd68 --pred_dir pretrain/div2k/pipeline/results_cbsd68_gpx2/dn+dm-sr/result --pred_pattern output
# python evaluate_metrics.py --test_dataset cbsd68 --pred_dir pretrain/div2k/pipeline/results_cbsd68_gpx2/dn+sr-dm/result --pred_pattern output

# # joint
# bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 12 div2k cbsd68 gp
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 div2k cbsd68 gp
# bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 12 div2k urban100 gp
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 div2k urban100 gp
#bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 12 div2k set14 gp
#bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 div2k set14 gp
#bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 12 div2k div2k
#bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 div2k div2k
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw rgb tenet rrdb 12 div2k urban100 g


# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw rgb tenet rrdb 12 div2k urban100 g
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw rgb tenet rrdb 12 div2k div2k g
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw rgb tenet rrdb 12 div2k cbsd68 g

# bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None rgb tenetv3 rrdb 12 div2k urban100 g
# bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None rgb tenetv3 rrdb 12 div2k div2k g
# bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None rgb tenetv3 rrdb 12 div2k cbsd68 g
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw rgb tenetv3 rrdb 12 div2k urban100 g
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw rgb tenetv3 rrdb 12 div2k div2k g
# bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw rgb tenetv3 rrdb 12 div2k cbsd68 g


# Now test on DIV2K
## sequential
#bash script_pipeline_compare.sh dm-dn-sr 2 rrdb 12 div2k div2k
#bash script_pipeline_compare.sh dm-sr-dn 2 rrdb 12 div2k div2k
#bash script_pipeline_compare.sh sr-dm-dn 2 rrdb 12 div2k div2k
#bash script_pipeline_compare.sh sr-dn-dm 2 rrdb 12 div2k div2k
#bash script_pipeline_compare.sh dn-dm-sr 2 rrdb 12 div2k div2k
#bash script_pipeline_compare.sh dn-sr-dm 2 rrdb 12 div2k div2k
### partial joint
#bash script_pipeline_compare.sh dn-dm+sr 2 rrdb 12 div2k div2k
#bash script_pipeline_compare.sh dn+dm-sr 2 rrdb 12 div2k div2k
#bash script_pipeline_compare.sh dn+sr-dm 2 rrdb 12 div2k div2k
## joint
#bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 12 div2k div2k
#bash script_e2e_compare.sh 2 e2e-dn-dm+sr noisy_lr_raw lr_raw linrgb tenet rrdb 12 div2k div2k
#bash script_e2e_compare.sh 2 e2e-dn+dm-sr noisy_lr_raw lr_linrgb linrgb tenet rrdb 12 div2k div2k
#bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 12 div2k div2k

# Test on real shot
# sonya73_1.ARW
# ps200
#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
#--model tenet --block rrdb --n_blocks 12 \
#--pretrain pretrain/pixelshift/pipeline/noisy_lr_raw-raw-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000-20210325-091926-ec84eecf-2e34-4a4e-b030-a8f660c74390/checkpoint/noisy_lr_raw-raw-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/sonya73_1.ARW
#
#
#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type None --out_type linrgb --scale 2 \
#--model tenet --block rrdb --n_blocks 12 \
#--pretrain /home/qiang/codefiles/low_level/ISP/ispnet/pretrain/pixelshift/pipeline/noisy_lr_raw-None-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000-20210316-213620-e552b770-14ab-4c17-a4d6-1e7eecef6d95/checkpoint/noisy_lr_raw-None-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/sonya73_1.ARW

#
#
## div2k
#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
#--model tenet --block rrdb --n_blocks 12 \
#--pretrain pretrain/div2k/pipeline/noisy_lr_raw-raw-linrgb-tenet-div2k-rrdb-n6-SR2-C64-B32-Patch128-Epoch200-20210823-024137-be205d74-a9e3-4a79-80a5-c1991157770b/checkpoint/noisy_lr_raw-raw-linrgb-tenet-div2k-rrdb-n6-SR2-C64-B32-Patch128-Epoch200_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/sonya73_1.ARW
#
## Jdndmsr
#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
#--model jdndmsr --block rcab --n_blocks 2 \
#--pretrain pretrain/pixelshift/pipeline/noisy_lr_raw-raw-linrgb-jdndmsr-pixelshift-rcab-n1-SR2-C64-B256-Patch128-Epoch1000-20210821-235515-499b0c3b-073b-417b-91f2-198f0e5bc954/checkpoint/noisy_lr_raw-raw-linrgb-jdndmsr-pixelshift-rcab-n1-SR2-C64-B256-Patch128-Epoch1000_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/sonya73_1.ARW
#
#
## jdsr
#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
#--model jdsr --block res --n_blocks 12 \
#--pretrain pretrain/pixelshift/pipeline/noisy_lr_raw-raw-linrgb-jdsr-pixelshift-res-n12-SR2-C64-B256-Patch128-Epoch1000-20210821-143138-53f725dc-c901-46d5-a5b4-e654625d0e43/checkpoint/noisy_lr_raw-raw-linrgb-jdsr-pixelshift-res-n12-SR2-C64-B256-Patch128-Epoch1000_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/sonya73_1.ARW

# dcraw.
#python test_realshot.py --phase test --in_type lr_linrgb --mid_type None --out_type linrgb --scale 2 \
#--model resnet --block rrdb --n_blocks 12 \
#--pretrain /home/qiang/codefiles/low_level/ISP/ispnet/pretrain/pixelshift/pipeline/lr_linrgb-None-linrgb-resnet-pixelshift-rrdb-n6-C64-SR2-B256-Patch128-Epoch1000-20210702-153713-caf0b6e5-9064-4625-bf74-038a337ecf04/checkpoint/lr_linrgb-None-linrgb-resnet-pixelshift-rrdb-n6-C64-SR2-B256-Patch128-Epoch1000_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/sonya73_1.ARW

#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
#--model tenet --block rrdb --n_blocks 12 \
#--pretrain pretrain/div2k/pipeline/noisy_lr_raw-raw-linrgb-tenet-div2k-rrdb-n6-SR2-C64-B32-Patch128-Epoch200-20210823-024137-be205d74-a9e3-4a79-80a5-c1991157770b/checkpoint/noisy_lr_raw-raw-linrgb-tenet-div2k-rrdb-n6-SR2-C64-B32-Patch128-Epoch200_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/KAUST_1.ARW --shot_noise 1.831155264e-05 --read_noise 3.07986151225e-07


# KAUST_2
#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
#--model tenet --block rrdb --n_blocks 12 \
#--pretrain pretrain/pixelshift/pipeline/noisy_lr_raw-raw-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000-20210325-091926-ec84eecf-2e34-4a4e-b030-a8f660c74390/checkpoint/noisy_lr_raw-raw-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/KAUST_2.dng 	--shot_noise 1.831155264e-05 --read_noise 3.07986151225e-07
#
#
#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type None --out_type linrgb --scale 2 \
#--model tenet --block rrdb --n_blocks 12 \
#--pretrain /home/qiang/codefiles/low_level/ISP/ispnet/pretrain/pixelshift/pipeline/noisy_lr_raw-None-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000-20210316-213620-e552b770-14ab-4c17-a4d6-1e7eecef6d95/checkpoint/noisy_lr_raw-None-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/KAUST_2.dng --shot_noise 1.831155264e-05 --read_noise 3.07986151225e-07

# # div2k
# python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
# --model tenet --block rrdb --n_blocks 12 \
# --pretrain pretrain/div2k/pipeline/noisy_lr_raw-raw-linrgb-tenet-div2k-rrdb-n6-SR2-C64-B32-Patch128-Epoch200-20210823-024137-be205d74-a9e3-4a79-80a5-c1991157770b/checkpoint/noisy_lr_raw-raw-linrgb-tenet-div2k-rrdb-n6-SR2-C64-B32-Patch128-Epoch200_checkpoint_best.pth \
# --save_dir data/testdata/result \
# --test_data data/testdata/original/DNG/KAUST_2.dng --shot_noise 1.831155264e-05 --read_noise 3.07986151225e-07

# # Jdndmsr
# python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
# --model jdndmsr --block rcab --n_blocks 2 \
# --pretrain pretrain/pixelshift/pipeline/noisy_lr_raw-raw-linrgb-jdndmsr-pixelshift-rcab-n1-SR2-C64-B256-Patch128-Epoch1000-20210821-235515-499b0c3b-073b-417b-91f2-198f0e5bc954/checkpoint/noisy_lr_raw-raw-linrgb-jdndmsr-pixelshift-rcab-n1-SR2-C64-B256-Patch128-Epoch1000_checkpoint_best.pth \
# --save_dir data/testdata/result \
# --test_data data/testdata/original/DNG/KAUST_2.dng --shot_noise 1.831155264e-05 --read_noise 3.07986151225e-07

# # jdsr
# python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
# --model jdsr --block res --n_blocks 12 \
# --pretrain pretrain/pixelshift/pipeline/noisy_lr_raw-raw-linrgb-jdsr-pixelshift-res-n12-SR2-C64-B256-Patch128-Epoch1000-20210821-143138-53f725dc-c901-46d5-a5b4-e654625d0e43/checkpoint/noisy_lr_raw-raw-linrgb-jdsr-pixelshift-res-n12-SR2-C64-B256-Patch128-Epoch1000_checkpoint_best.pth \
# --save_dir data/testdata/result \
# --test_data data/testdata/original/DNG/KAUST_2.dng --shot_noise 1.831155264e-05 --read_noise 3.07986151225e-07
