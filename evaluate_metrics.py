import os
import os.path as osp 
import argparse
from tqdm import tqdm
from glob import glob
import skimage
import cv2 
import numpy as np 
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.Metrics import cal_ssim, cal_freqgain, cal_fsim, freqgain


def main():
    gt_list = sorted(glob(osp.join(args.gt_dir, '*png')))
    pred_list = sorted(glob(osp.join(args.pred_dir, '*'+args.pred_pattern+'*png')))
    psnrs, ssims, freqgains, fsims = [], [], [], []
    # pbar = tqdm(enumerate(gt_list))
    for i, gt_path in enumerate(gt_list):
        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(cv2.imread(pred_list[i]), cv2.COLOR_BGR2RGB)
        H, W, C = pred.shape
        gt = gt[:H, :W]
        # print(i, pred.shape, gt.shape, pred_list[i], gt_path)
        psnr = skimage.metrics.peak_signal_noise_ratio(gt, pred)
        ssim = cal_ssim(gt, pred)
        freqgain = cal_freqgain(gt, pred)
        # fsim = cal_fsim(gt, pred)
        # freqgain, fsim = 0, 0
        psnrs.append(psnr)
        ssims.append(ssim)
        freqgains.append(freqgain)
        # fsims.append(fsim)
        print(f'{i+1:03d}, pred {osp.basename(pred_list[i])}, gt {osp.basename(gt_path)}'
              f' PSNR= {psnr} SSIM= {ssim} FreqGain= {freqgain}')
    avg_psnr = np.array(psnrs).mean()
    avg_ssim = np.array(ssims).mean()
    avg_freqgain = np.array(freqgains).mean()
    # avg_fsim = np.array(fsims).mean()
    print(f"Finish evaluating for {args.pred_dir}\n"
          f"PSNR= {avg_psnr} SSIM= {avg_ssim} FreqGain= {avg_freqgain}" 
          )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args
    main()