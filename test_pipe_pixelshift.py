import os
import argparse
import importlib
import torch
import pathlib
from torch.utils.data import DataLoader
from torchvision import utils
import torchvision.transforms.functional as TF
import scipy.io as sio
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.DataTools.FileTools import save_tensor_to_cv2img
from TorchTools.model_util import load_pretrained_models
from datasets import process
from datasets.generate_benchmark import LoadBenchmarkPixelShift, LoadBenchmark
from tqdm import tqdm


def main():
    # print('===> Loading the network ...')
    module = importlib.import_module("model.{}".format(args.model))
    model = module.Net(**vars(args)).to(args.device)

    # load pre-trained
    load_pretrained_models(model, args.pretrain)

    # -------------------------------------------------
    # test benchmark dataset
    if 'pixelshift' in args.test_dataset.lower():
        test_set = LoadBenchmarkPixelShift(args.benchmark_path,
                                           args.downsampler, args.scale,
                                           )
    else:
        test_set = LoadBenchmark(args.benchmark_path,
                                 args.downsampler, args.scale,
                                 noise_model=args.noise_model, sigma=args.sigma
                                 )
    test_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1,
                             shuffle=False, pin_memory=True)
    model.eval()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            # train, data convert
            if args.intermediate:
                raw_image_tmp = sio.loadmat(os.path.join(args.pre_dir, '%03d.mat' % (i + 1)))
                src_img = torch.unsqueeze(TF.to_tensor(raw_image_tmp['img_out']), dim=0).to(torch.float).to(args.device)
            else:
                src_img = data['noisy_lr_raw'].to(args.device)

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

            save_tensor_to_cv2img(rgb_in, os.path.join(args.save_dir, '%03d_input.png' % (i + 1)))
            save_tensor_to_cv2img(rgb_out, os.path.join(args.save_dir, '%03d_output.png' % (i + 1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args
    args.pre_dir = os.path.join(args.save_dir, "result-{}".format(args.pre_pipename))
    if args.intermediate:
        print("===> loading input data from results of : ", args.pre_dir)
    else:
        print("===> loading input data from  of : ", args.benchmark_path)

    # parse the desired pre-trained model from candidates
    print(f"===> try to find the pre-trained ckpt for {args.expprefix} in folder {args.pretrain}")
    path_file = None
    for root, dirs, files in os.walk(args.pretrain):
        for file in files:
            if file.startswith(args.pipename) and file.endswith("checkpoint_best.pth") and not ('lr' in args.in_type and f'SR{args.scale}' not in file):
                path_file = os.path.join(root, file)
    assert path_file is not None, "cannot find a checkpoint file"
    args.pretrain = path_file
    print(f"===> load pre-trained ckpt {args.pretrain}")

    print(args.save_dir)
    args.save_dir = os.path.join(args.save_dir, "result-{}".format(args.pipename))
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print("===> save results to : ", args.save_dir)
    main()
