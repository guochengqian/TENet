import os
import pdb
import shutil
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from TorchTools.DataTools.FileTools import _all_images
import numpy as np


def main(datasets_path, main_path, sigma=10):
    print('==============> configuration')
    # task configuration
    demo = 'python -u test.py ' \
           ' --pretrained_model checkpoints/ablation_study/demo-df2kx6-6-3-64-2-rrdb_checkpoint_1096.0k.path' \
           ' --model demo --block_type rrdb --n_resblocks 6 --channels 64 '\
           ' --datatype uint8 --crop_scale 1 '
    demo_deno = 'python -u test.py ' \
                ' --pretrained_model checkpoints/ablation_study/demo-dn-df2kx6-6-3-64-2-rrdb_checkpoint_1340.0k.path' \
                ' --model demo --block_type rrdb --n_resblocks 6 --channels 64 ' \
                ' --datatype uint8 --crop_scale 1  --denoise --sigma ' + str(sigma)
    denoraw = 'python -u test.py ' \
        ' --pretrained_model checkpoints/ablation_study/denoraw-dn-df2kx6-6-3-64-2-rrdb_checkpoint_1000.0k.path'\
        ' --model denoraw --block_type rrdb --n_resblocks 6 --channels 64 --bias '\
        ' --datatype uint8 --crop_scale 1 --denoise --sigma '+ str(sigma)

    denorgb = 'python -u test.py ' \
        ' --pretrained_model checkpoints/ablation_study/denorgb-dn-df2kx6-6-3-64-2-rrdb_checkpoint_830.0k.path'\
        ' --model denorgb --block_type rrdb --n_resblocks 6 --channels 64 '\
        ' --datatype uint8 --crop_scale 1 --denoise --sigma ' + str(sigma)

    srraw = 'python -u test.py ' \
        ' --pretrained_model checkpoints/ablation_study/srraw-df2kx6-6-3-64-2-rrdb_checkpoint_1000.0k.path'\
        ' --model srraw --block_type rrdb --n_resblocks 6 --channels 64 --bias '\
        ' --datatype uint8 --crop_scale 1 --scale 2'

    srraw_deno = 'python -u test.py ' \
        ' --pretrained_model checkpoints/ablation_study/srraw-dn-df2kx6-6-3-64-2-rrdb_checkpoint_1000.0k.path'\
        ' --model srraw --block_type rrdb --n_resblocks 6 --channels 64 --bias'\
        ' --datatype uint8 --crop_scale 1  --denoise --scale 2 --sigma ' + str(sigma)

    srrgb = 'python -u test.py ' \
        ' --pretrained_model checkpoints/ablation_study/srrgb-df2kx6-6-3-64-2-rrdb_checkpoint_1100.0k.path'\
        ' --model srrgb --block_type rrdb --n_resblocks 6 --channels 64 '\
        ' --datatype uint8 --crop_scale 1  --scale 2'

    srrgb_deno = 'python -u test.py ' \
        ' --pretrained_model checkpoints/ablation_study/srrgb-dn-df2kx6-6-3-64-2-rrdb_checkpoint_1790.0k.path'\
        ' --model srrgb --block_type rrdb --n_resblocks 6 --channels 64 '\
        ' --datatype uint8 --crop_scale 1 --scale 2 --denoise --sigma ' + str(sigma)

    tenet1 = 'python -u test.py ' \
        ' --pretrained_model checkpoints/ablation_study/tri1-df2kx6-6-6-64-2-rrdb_checkpoint_1000.0k.path'\
        ' --model tenet1 --block_type rrdb --sr_n_resblocks 6 --dm_n_resblocks 6 --scale 2 --bias --channels 64 '\
        ' --datatype uint8 --crop_scale 1 '

    tenet1_deno = 'python -u test.py ' \
        ' --pretrained_model checkpoints/ablation_study/tri1-dn-df2kx6-6-6-64-2-rrdb_checkpoint_890.0k.path'\
        ' --model tenet1 --block_type rrdb --sr_n_resblocks 6 --dm_n_resblocks 6 --scale 2 --bias --channels 64 '\
        ' --datatype uint8 --crop_scale 1 --denoise --sigma ' + str(sigma)

    tenet2_deno = 'python -u test.py ' \
        ' --pretrained_model checkpoints/ablation_study/tri2-dn-df2kx6-6-6-64-2-rrdb_checkpoint_1490.0k.path'\
        ' --model tenet2 --block_type rrdb --sr_n_resblocks 6 --dm_n_resblocks 6 --scale 2 --bias --channels 64 '\
        ' --datatype uint8 --crop_scale 1 --denoise --sigma ' + str(sigma)
    # add file location information
    # --test_path /data/sony/sim_test --save_path /data/sony/sim_test/kodak/psbx2/hr_rgb_denoise

    # comdlines dictionary
    cmdlines = {'demo': demo, 'demo_deno': demo_deno, 'denoraw': denoraw, 'denorgb': denorgb,
                'srraw': srraw, 'srraw_deno': srraw_deno, 'srrgb': srrgb, 'srrgb_deno': srrgb_deno,
                'tenet1': tenet1, 'tenet2_deno': tenet2_deno,  'tenet1_deno': tenet1_deno}

    # # # pipelines, multi-list
    pipelines = [['demo', 'denorgb', 'srrgb'], ['demo', 'srrgb', 'denorgb'], ['denoraw', 'demo', 'srrgb'],
                ['denoraw', 'srraw', 'demo'], ['srraw', 'demo', 'denorgb'], ['srraw', 'denoraw', 'demo'],
                ['demo_deno', 'srrgb'], ['srraw', 'demo_deno'], ['tenet1', 'denorgb'], ['denoraw', 'tenet1'],
                 ['demo', 'srrgb_deno'], ['srraw_deno', 'demo'], ['tenet1_deno'], ['tenet2_deno']]

    # datasets
    # datasets = ['kodak', 'mcm', 'kodak', 'urban100']
    datasets = ['mcm']
    # metrics
    cal_psnr = True
    psnrs = []
    ssims = []

    for dataset in datasets:
        src_path = os.path.join(datasets_path, 'noisy_input/sigma') + str(sigma) + '/' + dataset
        gt_path = os.path.join(datasets_path, 'gt/') + dataset

        temp_path1 = os.path.join(main_path, 'temp1')
        temp_path2 = os.path.join(main_path, 'temp2')
        temp_path3 = os.path.join(main_path, 'temp3')
        dst_path = os.path.join(main_path, 'result', dataset, 'final')
        # noise_path = '/data/sony/sim_test/kodak/psbx2/lr_input_sigma' + str(sigma)

        if not os.path.exists(temp_path1):
            os.makedirs(temp_path1)
        if not os.path.exists(temp_path2):
            os.makedirs(temp_path2)
        if not os.path.exists(temp_path3):
            os.makedirs(temp_path3)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

        # task input/output path
        input_path = [src_path, temp_path1, temp_path2]
        output_path = [temp_path1, temp_path2, temp_path3]

        # walk each possible pipeline
        for i, pipeline in enumerate(pipelines):
            # clear folder
            shutil.rmtree(temp_path1, ignore_errors=True)
            shutil.rmtree(temp_path2, ignore_errors=True)
            shutil.rmtree(temp_path3, ignore_errors=True)
            os.makedirs(temp_path1)
            os.makedirs(temp_path2)
            os.makedirs(temp_path3)

            # pdb.set_trace()
            # final_path = output_path[0]
            # do task

            new_folder = ''
            for j in range(len(pipeline)):
                task = pipeline[j]
                cmd = cmdlines[task] + ' --test_path ' + input_path[j] + ' --save_path ' + output_path[j]
                print(cmd)
                os.system(cmd)
                new_folder += task + '-'

            new_folder += 'final'
            new_dst_path = os.path.join(main_path, 'result', dataset, new_folder)
            if not os.path.exists(new_dst_path):
                os.makedirs(new_dst_path)

            if cal_psnr:
                gt_files = _all_images(gt_path)
                im_files = _all_images(output_path[j])
                gt_files.sort()
                im_files.sort()

                psnr = 0
                ssim = 0
                for k, img_path in enumerate(gt_files):
                    img_name = '/' + img_path.split('/')[-1].split('.')[0] + '_'
                    for im_file in im_files:
                        if img_name in im_file:
                            gt = imread(img_path)
                            img = imread(im_file)
                            w, h, c = img.shape
                            gt = gt[0:w, 0:h]
                            psnr += compare_psnr(gt, img)
                            ssim += compare_ssim(gt, img, data_range=255, multichannel=True)
                psnr /= k+1
                ssim /= k+1

                psnrs.append(psnr)
                ssims.append(ssim)
                print('pipeline{} psnr:{}  ssim:{}'.format(pipeline, psnr, ssim))


            for k in range(j):
                cmd = 'mv ' + os.path.join(output_path[k], '* ') + ' '+ new_dst_path
                os.system(cmd)
            cmd = 'cp ' + os.path.join(output_path[j], '* ') + ' ' + new_dst_path
            os.system(cmd)
            cmd = 'mv ' + os.path.join(output_path[j], '* ') + ' ' + dst_path
            os.system(cmd)

        print(psnrs)
        print(ssims)


if __name__ == '__main__':
    datasets_path = '/home/qiang/Documents/data/pixel-shift-200/TENet_sim_test'
    save_path = '/home/qiang/Documents/data/test/abalation'
    sigma = 10
    main(datasets_path, save_path, sigma)



