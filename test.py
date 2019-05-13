import os
import argparse
import importlib
import torch
import numpy as np
from TorchTools.ArgsTools.test_args import TestArgs
from TorchTools.DataTools.FileTools import _tensor2cvimage, _all_images, _read_image
from model.common import DownsamplingShuffle
import cv2
from model.common import print_model_parm_nums, demosaick_layer
import pdb
import torch.nn as nn

def main():
    ##############################################################################
    # args parse
    parser = argparse.ArgumentParser(description='PyTorch implementation of demosaicking')
    parsers = TestArgs()
    args = parsers.initialize(parser)
    if args.show_info:
        parsers.print_args()
    if args.datatype =='uint16':
        datatype = np.uint16
    else:
        datatype = np.uint8

    ##############################################################################
    # load models
    print('===> Loading the network ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(device)

    if args.show_info:
        print(model)
        print_model_parm_nums(model)

    ##############################################################################
    # pretrain
    if os.path.isfile(args.pretrained_model):
        print("=====> loading checkpoint '{}".format(args.pretrained_model))
        checkpoint = torch.load(args.pretrained_model)
        best_psnr = checkpoint['best_psnr']
        model.load_state_dict(checkpoint['state_dict'])
        print("The pretrained_model is at checkpoint {}k, and it's best loss is {}."
              .format(checkpoint['iter']/1000, best_psnr))
    else:
        print("=====> no checkpoint found at '{}'".format(args.pretrained_model))

    ##############################################################################

    if args.model == 'denorgb':
        in_channels = 3
        out_channels = 3
    elif args.model == 'denoraw':
        in_channels = 4
        out_channels = 1
    elif args.model == 'demo':
        in_channels = 4
        out_channels = 3
    elif args.model == 'srraw':
        in_channels = 4
        out_channels = 1
    elif args.model == 'srrgb':
        in_channels = 3
        out_channels = 3
    elif args.model == 'tri1':
        in_channels = 4
        out_channels = 3
    elif args.model == 'tri2':
        in_channels = 4
        out_channels = 3
    else:
        raise ValueError('not supported model')

    # test
    model.eval()
    raw_down_sample = DownsamplingShuffle(2)
    demosaic_layer = nn.PixelShuffle(2)
    # for datasets in os.listdir(args.test_path):
    img_path = os.path.join(args.test_path)
    dst_path = os.path.join(args.save_path)

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    im_files = _all_images(img_path)
    with torch.no_grad():
        for i in range(len(im_files)):
            im_file = im_files[i]
            paths = im_file.split('/')

            im_name = paths[-1]

            img = _read_image(im_file)
            if args.shift_x > 0:
                img = np.concatenate((img[:, 1:], img[:, -2:-1]), 1)
            if args.shift_y > 0:
                img = np.concatenate((img[1:], img[-2:-1]), 0)

            if in_channels == 4:
                img = torch.from_numpy(img).float().contiguous().view(-1, 1,
                                                                      img.shape[-2], img.shape[-1])
                img = raw_down_sample(img)
            else:
                img = torch.from_numpy(np.transpose(img, [2, 0, 1])).float()
                img = img.contiguous().view(-1, 3, img.shape[-2], img.shape[-1])

            if args.denoise:
                noise_map = torch.ones([1, 1, img.shape[-2], img.shape[-1]])*args.noise_level
                img = torch.cat((img, noise_map), 1)

            im_inputs = crop_imgs(img, args.crop_scale)
            del img

            im_inputs = im_inputs.to(device)

            # if input raw, but output rgb
            if in_channels != out_channels:
                output = torch.zeros((args.crop_scale ** 2, 1, out_channels, im_inputs.shape[2] * args.scale*2,
                                      im_inputs.shape[3] * args.scale*2))
            else:
                output = torch.zeros((args.crop_scale ** 2, 1, out_channels, im_inputs.shape[2] * args.scale,
                                      im_inputs.shape[3] * args.scale))

            for j in range(args.crop_scale ** 2):
                if args.model == 'tri2':
                    srraw, sr = model(im_inputs[j].unsqueeze(0))
                    del srraw
                else:
                    sr = model(im_inputs[j].unsqueeze(0))

                sr = torch.clamp(sr.cpu(), min=0., max=1.)
                output[j] = sr

            if args.crop_scale > 1:
                rgb_output = binning_imgs(output, args.crop_scale)
            else:
                rgb_output = output.view(1, output.shape[-3], output.shape[-2], output.shape[-1])

            if out_channels == 4:
                rgb_output = demosaic_layer(rgb_output)

            rgb_output = _tensor2cvimage(rgb_output, datatype)

            im_name = im_name.split('.')[0] + '-'+args.post
            # pdb.set_trace()
            cv2.imwrite(os.path.join(dst_path, im_name), rgb_output)
            if args.show_info:
                print('saving: {}, size: {} [{}]/[{}]'.format(os.path.join(dst_path, im_name),
                                                              rgb_output.shape, i, img_num-1))


def crop_imgs(img, ratio):
    b, c, h, w = img.shape
    h_psize = h // ratio
    w_psize = w // ratio
    imgs = torch.zeros(ratio ** 2, c, h_psize, w_psize)

    for i in range(ratio):
        for j in range(ratio):
            imgs[i * ratio + j] = img[0, :, i * h_psize:(i + 1) * h_psize, j * w_psize:(j + 1) * w_psize]
    return imgs


def binning_imgs(img, ratio):
    n, b, c, h, w = img.shape
    output = torch.zeros((b, c, h * ratio, w * ratio))

    for i in range(ratio):
        for j in range(ratio):
            output[:, :, i * h:(i + 1) * h, j * w:(j + 1) * w] = img[i * ratio + j]
    return output


if __name__ == '__main__':
     main()

