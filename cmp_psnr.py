import os
import pdb
import shutil
from skimage.measure import compare_psnr, compare_ssim
from skimage.io import imread, imsave
from TorchTools.DataTools.FileTools import _all_images
import numpy as np


def main():
    print('==============> PSNR Compare ICCV2019')
    # comdlines dictionary
    # methods = ['admm_edsr', 'condak_edsr', 'flexisp_edsr', 'demosaicknet_edsr', 'demosaicnet_carn', 'tri2_dn']
    # demos = ['admmn', 'condat', 'flexisp']
    demos = ['tri2']
    # demos = ['nlm', 'nat']
    # demos = ['matlab']
    # srs = ['edsr', 'carn']
    srs=['']
    # metrics
    psnrs = []
    ssims = []
    # datasets
    datasets = [ 'kodak', 'mcm', 'bsd300','urban100']

    # shift
    x_pad = 10
    y_pad = 10
    # path configuration
    main_folder = '/data/datasets/RawSR20181001/sim_test/'
    output_folder = os.path.join(main_folder, 'output/sr_out')
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
                    if sr !='':
                        img_name = gt_path.split('/')[-1].split('.')[0] + '_psbx2_'  + demo+'-'+sr +'.png'
                    else:
                        img_name = gt_path.split('/')[-1].split('.')[0] + '_psbx2-'  + demo+'.png'
                    # pdb.set_trace()
                    img_path = os.path.join(output_path, img_name)
                    img = imread(img_path)

                    w, h,c= img.shape
                    gt = gt[0:w, 0:h]

                    if x_pad != 0:
                        gt = gt[y_pad : w-y_pad, x_pad:h-x_pad]
                        img = img[y_pad : w-y_pad, x_pad:h-x_pad]
                    psnr += compare_psnr(gt, img)
                    ssim += compare_ssim(gt, img, data_range=255, multichannel=True)
                psnr /= k+1
                ssim /= k+1

                psnrs.append(psnr)
                ssims.append(ssim)

                print('dataset:{} demo:{} sr:{} psnr:{}  ssim:{}'.format(dataset, demo, sr, psnr, ssim))

    print('psnr:{}  ssim:{}'.format(psnrs, ssims))

if __name__ == '__main__':
    main()

#[26.82021657762418, 26.127086833773713, 28.291192662689422, 28.352354809563224, 26.55998350155382, 27.27737060600308, 28.35628323662874, 27.079588844039858, 28.452422943081846, 28.42365100053013]
#[0.672683424373921, 0.6235569766926038, 0.7997048510166963, 0.8013336110636118, 0.6542921088923472, 0.7116116571799366, 0.80189088610075, 0.6912227739372292, 0.80282474676964, 0.8026633091559973]





