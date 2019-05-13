import os
import pdb
import shutil
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from TorchTools.DataTools.FileTools import _all_images
import numpy as np


def main(sigma):
    print('==============> PSNR Compare ICCV2019')
    # comdlines dictionary
    # methods = ['admm_edsr', 'condak_edsr', 'flexisp_edsr', 'demosaicknet_edsr', 'demosaicnet_carn', 'tri2_dn']
    # demos = ['admmn', 'condat', 'flexisp']
    # demos = ['mit']
    # demos = ['admm', 'condat', 'flexisp', 'mit']
    demos = ['mit']
    srs = ['carn-joint']
    # metrics
    psnrs = []
    ssims = []
    # datasets
    datasets = [ 'kodak', 'mcm', 'bsd300','urban100']
    # datasets = ['urban100']
    # path configuration
    main_folder = '/data/datasets/RawSR20181001/sim_test/'
    output_folder = os.path.join(main_folder, 'output/sr_dn_out')
    gt_folder = os.path.join(main_folder, 'gt')

    for dataset in datasets:
        output_path = os.path.join(output_folder, dataset, 'psbx2')
        gt_path = os.path.join(gt_folder, dataset)

        gt_files = _all_images(gt_path)
        for demo in demos:
            for sr in srs:
                psnr = 0
                ssim = 0
                for k, gt_path in enumerate(gt_files):
                    gt = imread(gt_path)
                    img_name = gt_path.split('/')[-1].split('.')[0] + '_psbx2_'  + demo+ str(sigma)+'-'+sr +'.png'
                    # img_name = gt_path.split('/')[-1].split('.')[0] + '_psbx2-sigma' + str(sigma)+ '-'+ demo+'.png'
                    img_path = os.path.join(output_path, img_name)
                    img = imread(img_path)

                    w, h,c= img.shape
                    gt = gt[0:w, 0:h]
                    psnr += compare_psnr(gt, img)
                    ssim += compare_ssim(gt, img, data_range=255, multichannel=True)
                psnr /= k+1
                ssim /= k+1

                psnrs.append(psnr)
                ssims.append(ssim)

                print('sigma{}: dataset:{} demo:{} sr:{} psnr:{}  ssim:{}'.format(sigma, dataset, demo, sr, psnr, ssim))

    print('sigma{} psnr:{}  ssim:{}'.format(sigma, psnrs, ssims))

if __name__ == '__main__':
    # sigmas = [10, 20, 50]
    sigmas = [10, 20]
    for sigma in sigmas:
        main(sigma)

#[26.82021657762418, 26.127086833773713, 28.291192662689422, 28.352354809563224, 26.55998350155382, 27.27737060600308, 28.35628323662874, 27.079588844039858, 28.452422943081846, 28.42365100053013]
#[0.672683424373921, 0.6235569766926038, 0.7997048510166963, 0.8013336110636118, 0.6542921088923472, 0.7116116571799366, 0.80189088610075, 0.6912227739372292, 0.80282474676964, 0.8026633091559973]





