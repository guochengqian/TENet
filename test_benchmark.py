import os
import argparse
import importlib
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import scipy.io as sio
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.DataTools.FileTools import save_image_tensor2cv2
from datasets import process
from datasets.generate_benchmark import LoadBenchmark, LoadBenchmarkPixelShift
from model.common import cal_model_parm_nums
from TorchTools.model_util import load_pretrained_models
from tqdm import tqdm


def main():
    print('===> Loading the network ...')
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(args.device)
    # print(model)
    model_size = cal_model_parm_nums(model)
    print('Number of params: %.4f M' % (model_size / 1e6))

    # load pre-trained
    model, best_psnr, epoch = load_pretrained_models(model, args.pretrain)

    # -------------------------------------------------
    # load benchmark dataset
    dataset = os.path.basename(os.path.dirname(args.benchmark_path))
    print(f'===> loading benchmark dataset {dataset} from path {args.benchmark_path}')

    if 'pixelshift' in dataset.lower():
        test_set = LoadBenchmarkPixelShift(args.benchmark_path,
                                           args.downsampler, args.scale,
                                           args.in_type, args.mid_type, args.out_type
                                           )
    else:
        test_set = LoadBenchmark(args.benchmark_path,
                                 args.downsampler, args.scale,
                                 args.in_type, args.mid_type, args.out_type
                                 )
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1,
                             shuffle=False, pin_memory=True)

    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            # train, data convert
            src_img = data[args.in_type].to(args.device)

            if 'noisy' in args.in_type:
                shot_noise = data['noise']['shot_noise'].to(args.device)
                read_noise = data['noise']['read_noise'].to(args.device)
                variance = shot_noise * src_img + read_noise
                img_in = torch.cat((src_img, variance), dim=1)
            else:
                img_in = src_img

            img_out = model(img_in)
            img_out_cpu = img_out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

            # save mat
            sio.savemat(os.path.join(args.save_dir, '%03d.mat' % (i + 1)), {'img_out': img_out_cpu})

            # Post-Processing for saving the results
            red_g, blue_g, ccm = data['metadata']['red_gain'].to(args.device), \
                                 data['metadata']['blue_gain'].to(args.device), \
                                 data['metadata']['ccm'].to(args.device)

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

            save_image_tensor2cv2(rgb_in, os.path.join(args.save_dir, '%03d_input.png' % (i + 1)))
            save_image_tensor2cv2(rgb_out, os.path.join(args.save_dir, '%03d_output.png' % (i + 1)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args

    # parse the desired pre-trained model from candidates
    print(f"===> try to find the pre-trained ckpt for {args.exp_prefix}")
    path_file = None
    for root, dirs, files in os.walk(args.pretrain):
        for file in files:
            if file.startswith(args.exp_prefix) and f'SR{args.scale}' in file and file.endswith("checkpoint_best.pth"):
                path_file = os.path.join(root, file)
    assert path_file is not None, "cannot find a checkpoint file"
    args.pretrain = path_file
    print(f"===> load pre-trained ckpt {args.pretrain}")
    main()
