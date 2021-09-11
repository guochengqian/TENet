import os
import argparse
import h5py
import numpy as np
import importlib
import torch
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
from TorchTools.model_util import load_pretrained_models
from TorchTools.ArgsTools.base_args import BaseArgs
from datasets import process
from model.common import cal_model_parm_nums


def main():
    print('===> Loading the network ...')
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(args.device)
    print(model)
    model_size = cal_model_parm_nums(model)
    print('Number of params: %.4f Mb' % (model_size / 1e6))

    # load pre-trained
    model, _, _ = load_pretrained_models(model, args.pretrain)

    # test DnD dataset
    # Loads image information and bounding boxes.
    info = h5py.File(os.path.join(args.test_data, 'info.mat'), 'r')['info']
    bb = info['boundingboxes']

    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]])
    # Denoise each image.
    for i in range(0, 50):
        # Loads the noisy image.
        filename = os.path.join(args.test_data, 'images_raw', '%04d.mat' % (i + 1))
        print('Processing file: %s' % filename)
        with h5py.File(filename, 'r') as img:
            noisy = np.float32(np.array(img['Inoisy']).T)

        # Loads raw Bayer color pattern.
        bayer_pattern = np.asarray(info[info['camera'][0][i]]['pattern']).tolist()
        # Load the camera's (or image's) ColorMatrix2
        xyz2cam = torch.FloatTensor(np.reshape(np.asarray(info[info['camera'][0][i]]['ColorMatrix2']), (3, 3)))
        # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
        rgb2cam = torch.mm(xyz2cam, rgb2xyz)
        # Normalizes each row.
        rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
        cam2rgb = torch.inverse(rgb2cam)

        # Specify red and blue gains here (for White Balancing)
        asshotneutral = info[info['camera'][0][i]]['AsShotNeutral']
        red_gain = torch.FloatTensor(asshotneutral[1] / asshotneutral[0])
        blue_gain = torch.FloatTensor(asshotneutral[1] / asshotneutral[2])

        ccm = torch.unsqueeze(cam2rgb, dim=0)
        red_g = torch.unsqueeze(red_gain, dim=0)
        blue_g = torch.unsqueeze(blue_gain, dim=0)

        # Denoise each bounding box in this image.
        boxes = np.array(info[bb[0][i]]).T
        for k in range(20):
            # Crops the image to this bounding box.
            idx = [
                int(boxes[k, 0] - 1),
                int(boxes[k, 2]),
                int(boxes[k, 1] - 1),
                int(boxes[k, 3])
            ]
            noisy_crop = noisy[idx[0]:idx[1], idx[2]:idx[3]].copy()

            # Flips the raw image to ensure RGGB Bayer color pattern.
            if bayer_pattern == [[1, 2], [2, 3]]:
                pass
            elif bayer_pattern == [[2, 1], [3, 2]]:
                noisy_crop = np.fliplr(noisy_crop)
            elif bayer_pattern == [[2, 3], [1, 2]]:
                noisy_crop = np.flipud(noisy_crop)
            else:
                print('Warning: assuming unknown Bayer pattern is RGGB.')

            # Loads shot and read noise factors.
            nlf_h5 = info[info['nlf'][0][i]]
            shot_noise = nlf_h5['a'][0][0]
            read_noise = nlf_h5['b'][0][0]

            # Extracts each Bayer image plane.
            height, width = noisy_crop.shape
            noisy_bayer = []
            for yy in range(2):
                for xx in range(2):
                    noisy_crop_c = noisy_crop[yy:height:2, xx:width:2].copy()
                    noisy_bayer.append(noisy_crop_c)
            noisy_bayer = np.stack(noisy_bayer, axis=-1)
            variance = shot_noise * noisy_bayer + read_noise

            raw_image_in = torch.unsqueeze(TF.to_tensor(noisy_bayer), dim=0).to(torch.float).cuda()
            raw_image_var = torch.unsqueeze(TF.to_tensor(variance), dim=0).to(torch.float).cuda()

            # Image ISP Here
            model.eval()
            with torch.no_grad():
                if 'noisy' in args.in_type:
                    model_input = torch.cat((raw_image_in, raw_image_var), dim=1)
                else:
                    model_input = raw_image_in
                model_out = model(model_input)

            noisy_bayer = raw_image_in.cpu()
            model_out = model_out.cpu()

            # Post-Processing for saving the results
            noisy_rgb = process.raw2srgb(noisy_bayer, red_g, blue_g, ccm)
            if 'raw' in args.out_type:
                denoised_rgb = process.raw2srgb(model_out, red_g, blue_g, ccm)
            elif 'linrgb' in args.out_type:
                denoised_rgb = process.rgb2srgb(model_out, red_g, blue_g, ccm)

            utils.save_image(noisy_rgb, os.path.join(args.save_dir, '%04d_%02d_noisy.png' % (i + 1, k + 1)))
            utils.save_image(denoised_rgb, os.path.join(args.save_dir, '%04d_%02d_%s.png' % (i + 1, k + 1, args.model)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args
    main()
