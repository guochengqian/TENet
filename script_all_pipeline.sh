#!/usr/bin/env bash
# shopt -s extglob

# Test on pixelshift
## sequential
### SR 2
#bash script_pipeline_compare.sh dm-dn-sr 2 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh dm-sr-dn 2 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh sr-dm-dn 2 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh sr-dn-dm 2 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh dn-dm-sr 2 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh dn-sr-dm 2 rrdb 6 pixelshift pixelshift

## partial joint
#bash script_pipeline_compare.sh dn-dm+sr 2 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh dn+dm-sr 2 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh dn+sr-dm 2 rrdb 6 pixelshift pixelshift

## joint
#bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 6 pixelshift pixelshift
#bash script_e2e_compare.sh 2 e2e-dn-dm+sr noisy_lr_raw lr_raw linrgb tenet rrdb 6 pixelshift pixelshift
#bash script_e2e_compare.sh 2 e2e-dn+dm-sr noisy_lr_raw lr_linrgb linrgb tenet rrdb 6 pixelshift pixelshift
#bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 6 pixelshift pixelshift


### SR 4
#bash script_pipeline_compare.sh dm-dn-sr 4 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh dm-sr-dn 4 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh sr-dm-dn 4 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh sr-dn-dm 4 rrdb 6 pixelshift pixelshift
## bash script_pipeline_compare.sh dn-dm-sr 4 rrdb 6 pixelshift pixelshift
## bash script_pipeline_compare.sh dn-sr-dm 4 rrdb 6 pixelshift pixelshift

#bash script_pipeline_compare.sh dn-dm+sr 4 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh dn+dm-sr 4 rrdb 6 pixelshift pixelshift
#bash script_pipeline_compare.sh dn+sr-dm 4 rrdb 6 pixelshift pixelshift

#bash script_e2e_compare.sh 4 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 6 pixelshift pixelshift
#bash script_e2e_compare.sh 4 e2e-dn-dm+sr noisy_lr_raw lr_raw linrgb tenet rrdb 6 pixelshift pixelshift
#bash script_e2e_compare.sh 4 e2e-dn+dm-sr noisy_lr_raw lr_linrgb linrgb tenet rrdb 6 pixelshift pixelshift
#bash script_e2e_compare.sh 4 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 6 pixelshift pixelshift


## E2E using other modules.
### Res
#bash script_e2e_compare.sh 2 resnet-dn+dm+sr noisy_lr_raw None linrgb tenet res 6  pixelshift pixelshift
#bash script_e2e_compare.sh 2 resnet-dn+sr-dm noisy_lr_raw raw linrgb tenet res 6  pixelshift pixelshift
#
### NLSA
#bash script_e2e_compare.sh 2 nlsa-dn+dm+sr noisy_lr_raw None linrgb tenet nlsa 6 pixelshift pixelshift
#bash script_e2e_compare.sh 2 nlsa-dn+sr-dm noisy_lr_raw raw linrgb tenet nlsa 6 pixelshift pixelshift


# Ablate dataset
## sequential
#bash script_pipeline_compare.sh dm-dn-sr 2 rrdb 6 div2k urban100
#bash script_pipeline_compare.sh dm-sr-dn 2 rrdb 6 div2k urban100
#bash script_pipeline_compare.sh sr-dm-dn 2 rrdb 6 div2k urban100
#bash script_pipeline_compare.sh sr-dn-dm 2 rrdb 6 div2k urban100
#bash script_pipeline_compare.sh dn-dm-sr 2 rrdb 6 div2k urban100
#bash script_pipeline_compare.sh dn-sr-dm 2 rrdb 6 div2k urban100
### partial joint
#bash script_pipeline_compare.sh dn-dm+sr 2 rrdb 6 div2k urban100
#bash script_pipeline_compare.sh dn+dm-sr 2 rrdb 6 div2k urban100
#bash script_pipeline_compare.sh dn+sr-dm 2 rrdb 6 div2k urban100

## joint
#bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 6 div2k urban100
#bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 6 div2k urban100

#bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 6 div2k cbsd68
#bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 6 div2k cbsd68
#
#bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 6 div2k set14
#bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 6 div2k set14
#
#bash script_e2e_compare.sh 2 e2e-dn+dm+sr noisy_lr_raw None linrgb tenet rrdb 6 div2k div2k
#bash script_e2e_compare.sh 2 e2e-dn+sr-dm noisy_lr_raw raw linrgb tenet rrdb 6 div2k div2k


# Test on real shot
# sonya73_1.ARW
# ps200
#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
#--model tenet --block_type rrdb --n_blocks 6 \
#--pretrain pretrain/pixelshift/pipeline/noisy_lr_raw-raw-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000-20210325-091926-ec84eecf-2e34-4a4e-b030-a8f660c74390/checkpoint/noisy_lr_raw-raw-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/sonya73_1.ARW

#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type None --out_type linrgb --scale 2 \
#--model tenet --block_type rrdb --n_blocks 6 \
#--pretrain /home/qiang/codefiles/low_level/ISP/ispnet/pretrain/pixelshift/pipeline/noisy_lr_raw-None-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000-20210316-213620-e552b770-14ab-4c17-a4d6-1e7eecef6d95/checkpoint/noisy_lr_raw-None-linrgb-tenet-pixelshift-rrdb-n6-C64-SR2-B32-Patch128-Epoch1000_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/sonya73_1.ARW

## div2k
#python test_realshot.py --phase test --in_type noisy_lr_raw --mid_type raw --out_type linrgb --scale 2 \
#--model tenet --block_type rrdb --n_blocks 6 \
#--pretrain pretrain/div2k/pipeline/noisy_lr_raw-raw-linrgb-tenet-div2k-rrdb-n6-SR2-C64-B32-Patch128-Epoch200-20210823-024137-be205d74-a9e3-4a79-80a5-c1991157770b/checkpoint/noisy_lr_raw-raw-linrgb-tenet-div2k-rrdb-n6-SR2-C64-B32-Patch128-Epoch200_checkpoint_best.pth \
#--save_dir data/testdata/result \
#--test_data data/testdata/original/DNG/sonya73_1.ARW

