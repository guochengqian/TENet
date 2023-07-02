import os
import argparse
import h5py
import numpy as np
import importlib
import torch
import pathlib
import scipy.io as sio
from datasets import process
from torchvision import transforms, utils
from TorchTools.ArgsTools.pipe_args import BaseArgs
from TorchTools.model_util import load_pretrained_models


def main():
    # print('===> Loading the network ...')
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(args.device)

    ##############################################################################
    # load pre-trained
    model, _, _ = load_pretrained_models(model, args.pretrain)

    # -------------------------------------------------
    # test DnD dataset
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]])
    totensor_ = transforms.ToTensor()

    # Loads image information and bounding boxes.
    info = h5py.File(os.path.join(args.test_data, 'info.mat'), 'r')['info']
    bb = info['boundingboxes']

    # Denoise each image.
    for i in range(0, 50):
        if not args.intermediate:
            # Loads the noisy image.
            filename = os.path.join(args.test_data, 'images_raw', '%04d.mat' % (i + 1))
            print('Processing file: %s' % filename)
            if not args.intermediate:
                with h5py.File(filename, 'r') as img:
                    noisy = np.float32(np.array(img['Inoisy']).T)

            # Loads raw Bayer color pattern.
            bayer_pattern = np.asarray(info[info['camera'][0][i]]['pattern']).tolist()
            # Load the camera's (or image's) ColorMatrix2
            xyz2cam = torch.FloatTensor(np.reshape(np.asarray(info[info['camera'][0][i]]['ColorMatrix2']), (3, 3)))
            # print(bayer_pattern, xyz2cam)
            # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
            rgb2cam = torch.mm(xyz2cam, rgb2xyz)
            # Normalizes each row.
            rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
            cam2rgb = torch.inverse(rgb2cam)
            # print(cam2rgb, cam2rgb.size())
            # Specify red and blue gains here (for White Balancing)
            asshotneutral = info[info['camera'][0][i]]['AsShotNeutral']
            red_gain = torch.FloatTensor(asshotneutral[1] / asshotneutral[0])
            blue_gain = torch.FloatTensor(asshotneutral[1] / asshotneutral[2])
            # Post-Processing for saving the results
            ccm = torch.unsqueeze(cam2rgb, dim=0).cuda()
            red_g = torch.unsqueeze(red_gain, dim=0).cuda()
            blue_g = torch.unsqueeze(blue_gain, dim=0).cuda()

            # Denoises each bounding box in this image.
            boxes = np.array(info[bb[0][i]]).T

        for k in range(20):
            if not args.intermediate:
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
                height, width = noisy_crop.shape
                noisy_bayer = []
                for yy in range(2):
                    for xx in range(2):
                        noisy_crop_c = noisy_crop[yy:height:2, xx:width:2].copy()
                        noisy_bayer.append(noisy_crop_c)
                noisy_bayer = np.stack(noisy_bayer, axis=-1)

            # Loads shot and read noise factors.
            nlf_h5 = info[info['nlf'][0][i]]
            shot_noise = nlf_h5['a'][0][0]
            read_noise = nlf_h5['b'][0][0]

            # Extracts each Bayer image plane.
            if args.intermediate:
                raw_image_tmp = sio.loadmat(os.path.join(args.pre_dir, '%04d_%02d.mat' % (i + 1, k + 1)))
                raw_image_tmp = np.float32(np.array(raw_image_tmp['model_out']))
                # print(np.shape(raw_image_tmp))
                # raw_image_in = torch.unsqueeze(totensor_(raw_image_tmp), dim=0).to(torch.float).cuda()
            else:
                raw_image_tmp = noisy_bayer
                # raw_image_in = torch.unsqueeze(totensor_(noisy_bayer), dim=0).to(torch.float).cuda()

            variance = shot_noise * raw_image_tmp + read_noise
            raw_image_in = torch.unsqueeze(totensor_(raw_image_tmp), dim=0).to(torch.float).cuda()
            # raw_image_var = torch.unsqueeze()
            raw_image_var = torch.unsqueeze(totensor_(variance), dim=0).to(torch.float).cuda()

            # Image ISP Here
            model.eval()
            with torch.no_grad():
                if 'noisy' in args.in_type:
                    model_input = torch.cat((raw_image_in, raw_image_var), dim=1)
                else:
                    model_input = raw_image_in
                model_out = model(model_input)
            model_read_out = model_out[0, :, :, :].detach().cpu().data.numpy().transpose(1,2,0)

            # save mat
            sio.savemat(os.path.join(args.save_dir, '%04d_%02d.mat' % (i + 1, k + 1)), {'model_out' : model_read_out})

            if 'raw' in args.pre_out_type:
                noisy_rgb = process.raw2srgb(raw_image_in, red_g, blue_g, ccm)
            elif 'linrgb' in args.pre_out_type:
                noisy_rgb = process.rgb2srgb(raw_image_in, red_g, blue_g, ccm)
            if 'raw' in args.out_type:
                denoised_rgb = process.raw2srgb(model_out, red_g, blue_g, ccm)
            elif 'linrgb' in args.out_type:
                denoised_rgb = process.rgb2srgb(model_out, red_g, blue_g, ccm)
            elif 'rgb' in args.out_type:
                denoised_rgb = process.rgb2srgb(model_out, red_g, blue_g, ccm)

            utils.save_image(noisy_rgb, os.path.join(args.save_dir, '%04d_%02d_noisy.png' % (i + 1, k + 1)))
            utils.save_image(denoised_rgb, os.path.join(args.save_dir, '%04d_%02d_%s.png' % (i + 1, k + 1, args.model)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args
    args.pre_dir = os.path.join(args.save_dir, "result-{}".format(args.pre_jobname))
    if args.intermediate:
        print("===> loading input data from results of : ", args.pre_dir)

    path_file = None
    for root, dirs, files in os.walk(args.pretrain):
        for file in files:
            if file.startswith(args.jobname) and file.endswith("checkpoint_best.pth"):
                path_file = os.path.join(root, file)
    assert path_file is not None, "cannot find a checkpoint file for {} in {}".format(args.jobname, args.pretrain)
    args.pretrain = path_file

    args.save_dir = os.path.join(args.save_dir, "result-{}".format(args.jobname))
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print("===> save results to : ", args.save_dir)
    main()
