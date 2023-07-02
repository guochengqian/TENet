import os
import os.path as osp 
import argparse
import importlib
from tqdm import tqdm
from glob import glob
import skimage
import cv2 
import numpy as np 
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import scipy.io as sio
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.DataTools.FileTools import tensor2np
from datasets import process
from datasets.generate_benchmark import LoadBenchmark, LoadBenchmarkPixelShift
from model.common import cal_model_parm_nums
from TorchTools.model_util import load_pretrained_models
from TorchTools.Metrics import cal_ssim, cal_freqgain, freqgain


@torch.no_grad()
def main():
    print('===> Loading the network ...')
    module = importlib.import_module("model.{}".format(args.model))
    model = module.Net(**vars(args)).to(args.device)
    # print(model)
    model_size = cal_model_parm_nums(model)
    print('Number of params: %.4f M' % (model_size / 1e6))

    # load pre-trained
    load_pretrained_models(model, args.pretrain)

    # -------------------------------------------------
    # load benchmark dataset
    dataset= args.dataset
    print(f'===> loading benchmark dataset {dataset} from path {args.benchmark_path}')

    if 'pixelshift' in dataset.lower():
        test_set = LoadBenchmarkPixelShift(args.benchmark_path,
                                           args.downsampler, args.scale,
                                           args.in_type, args.mid_type, args.out_type, 
                                           )
    else:
        test_set = LoadBenchmark(args.benchmark_path,
                                 args.downsampler, args.scale,
                                 args.in_type, args.mid_type, args.out_type, 
                                 noise_model=args.noise_model, sigma=args.sigma
                                 )
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1,
                             shuffle=False, pin_memory=True)

    model.eval()
    psnrs, ssims, freqgains, fsims = [], [], [], []
    for i, data in tqdm(enumerate(test_loader)):
        src_img = data[args.in_type].to(args.device)
        if 'noisy' in args.in_type:
            if 'p' in args.noise_model:
                shot_noise = data['noise']['shot_noise'].to(args.device)
                read_noise = data['noise']['read_noise'].to(args.device)
                variance = shot_noise * src_img + read_noise
            else:
                variance = data['variance'].to(args.device)
            img_in = torch.cat((src_img, variance), dim=1)
        else:
            img_in = src_img
        with torch.no_grad():
            img_out = model(img_in)

        if 'metadata' in data.keys():
            red_g, blue_g, ccm = data['metadata']['red_gain'].to(args.device), \
                                    data['metadata']['blue_gain'].to(args.device), \
                                    data['metadata']['ccm'].to(args.device)
        else:
            red_g, blue_g, ccm = None, None, None
        if 'raw' in args.out_type:
            rgb_out = process.raw2srgb(img_out, red_g, blue_g, ccm)
        elif 'linrgb' in args.out_type:
            rgb_out = process.rgb2srgb(img_out, red_g, blue_g, ccm)
        else:
            rgb_out = img_out

        if 'raw' in args.in_type:
            rgb_in = process.raw2srgb(src_img, red_g, blue_g, ccm)
        elif 'linrgb' in args.in_type:
            rgb_in = process.rgb2srgb(src_img, red_g, blue_g, ccm)
        else:
            rgb_in = src_img

        rgb_in = tensor2np(rgb_in)        
        rgb_out = tensor2np(rgb_out)        
        cv2.imwrite(os.path.join(args.save_dir, '%03d_input.png' % (i + 1)), cv2.cvtColor(rgb_in, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.save_dir, '%03d_output.png' % (i + 1)), cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR))
        # test PSNR and SSIM
        # load GT.
        # gt_path = [item for item in gt_list if '%03d' % (i + 1) in item][-1]
        # gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
        gt = tensor2np(data['rgb'])
        psnr = skimage.metrics.peak_signal_noise_ratio(gt, rgb_out)
        ssim = cal_ssim(gt, rgb_out)
        freqgain = cal_freqgain(gt, rgb_out)
        # fsim = cal_fsim(gt, rgb_out)
        # freqgain, fsim = 0, 0
        psnrs.append(psnr)
        ssims.append(ssim)
        freqgains.append(freqgain)
        # fsims.append(fsim)
        # print(f'{i+1:03d}, PSNR= {psnr} SSIM= {ssim} FreqGain= {freqgain} FSIM= {fsim}')
        print(f'{i+1:03d}, PSNR= {psnr} SSIM= {ssim} FreqGain= {freqgain}')
    avg_psnr = np.array(psnrs).mean()
    avg_ssim = np.array(ssims).mean()
    avg_freqgain = np.array(freqgains).mean()
    # avg_fsim = np.array(fsims).mean()
    print(f"Finish testing. Avg Results are saved to {args.save_dir}."
          f"PSNR= {avg_psnr} SSIM= {avg_ssim} FreqGain= {avg_freqgain} "
        #   f"FSIM= {avg_fsim}" 
          )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args

    # parse the desired pre-trained model from candidates
    print(f"===> try to find the pre-trained ckpt for {args.expprefix} in folder {args.pretrain}")
    path_file = None
    for root, dirs, files in os.walk(args.pretrain):
        for file in files:
            if file.startswith(args.expprefix) and f'SR{args.scale}' in file and file.endswith("_best.pth"):
                if 'p' in args.noise_model or ('noise_'+args.noise_model+'-') in file:
                    path_file = os.path.join(root, file)
    assert path_file is not None, "cannot find a checkpoint file"
    args.pretrain = path_file
    print(f"===> load pre-trained ckpt {args.pretrain}")
    main()