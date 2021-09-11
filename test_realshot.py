import os
import argparse
import numpy as np
import importlib
import torch

from TorchTools.DataTools.FileTools import save_image_tensor2cv2
from TorchTools.ArgsTools.pipe_args import BaseArgs
from TorchTools.model_util import load_pretrained_models
from datasets import process
import torchvision.transforms.functional as TF
import math

import rawpy
import rawpy.enhance


def process_single_raw(src_path):
    # src_path
    raw_file = rawpy.imread(src_path)
    black_level = np.mean(raw_file.black_level_per_channel)
    white_level = float(raw_file.white_level)

    # raw image
    if 'rgb' in args.in_type:
        image = raw_file.postprocess(gamma=(1, 1), no_auto_bright=True, no_auto_scale=False, output_bps=16, output_color=rawpy.ColorSpace.raw)
        image = image/65535. * white_level
    else:
        image = raw_file.raw_image.astype(np.float32)
    image = (image - black_level) / (white_level - black_level)
    image = np.clip(image, 0, 1.0)

    # metadata
    # colormatrix = np.asarray(raw_file.rgb_xyz_matrix[0:3, 0:3])
    colormatrix = np.asarray([[1.087177634, -0.4568284452, -0.1467799544], [-0.4628916085, 1.416070819, 0.2814568579],
      [-0.02583973669, 0.0994599387, 0.7301111817]]).astype(np.float32)

    # colormatrix = (raw_file.color_matrix[0:3, 0:3])
    gains = raw_file.daylight_whitebalance
    red_gain = np.asarray([gains[0] / gains[1]])
    blue_gain = np.asarray([gains[2] / gains[1]])

    metadata = {
        'colormatrix': colormatrix,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return image, metadata


def main():
    # # print('===> Loading the network ...')
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(args.device)

    ##############################################################################
    # load pre-trained
    model, _, _ = load_pretrained_models(model, args.pretrain)

    # read image
    rggb, matainfo = process_single_raw(args.test_data)
    ccm, red_g, blue_g = process.metadata2tensor(matainfo)
    ccm, red_g, blue_g = ccm.to(args.device), red_g.to(args.device), blue_g.to(args.device)

    raw_image_in = torch.unsqueeze(TF.to_tensor(rggb.astype(np.float32)), dim=0).to(args.device)
    # # ------- begin of debug
    #     linrgb = torch.stack((raw_image_in[:, 0, :, :], raw_image_in[:, 1, :, :] / 2 + raw_image_in[:, 2, :, :] / 2, raw_image_in[:, 3, :, :]), dim=1)
    #     srgb = process.rgb2srgb(linrgb, red_g, blue_g, ccm)
    # # --------- end of debug

    if 'raw' in args.in_type:
        B, C, H, W = raw_image_in.shape
        raw_image_in = raw_image_in.view(B, C, H//2, 2, W//2, 2).permute(0, 1, 3, 5, 2, 4).contiguous().view(B, 4, H//2, W//2)
        scale = args.scale * 2
    else:
        scale = args.scale

    # has to chunk the input image
    if 'noisy' in args.in_type:
        shot_noise = args.shot_noise
        read_noise = args.read_noise
        raw_image_var = shot_noise * raw_image_in + read_noise
        raw_image_in = torch.cat((raw_image_in, raw_image_var), dim=1)

    MAX_PATCH_SIZE = 512
    h, w = raw_image_in.shape[-2:]
    n_chunks_h = math.ceil(h/MAX_PATCH_SIZE)
    n_chunks_w = math.ceil(w/MAX_PATCH_SIZE)

    img_out = torch.zeros(1, 3, h*scale, w*scale)
    model.eval()
    for i_h in range(n_chunks_h):
        if (i_h+1) * MAX_PATCH_SIZE > h:
            h_start = h - MAX_PATCH_SIZE
            h_end = h
        else:
            h_start = i_h * MAX_PATCH_SIZE
            h_end = (i_h+1) * MAX_PATCH_SIZE

        for i_w in range(n_chunks_w):
            if (i_w + 1) * MAX_PATCH_SIZE > w:
                w_start = w - MAX_PATCH_SIZE
                w_end = w
            else:
                w_start = i_w * MAX_PATCH_SIZE
                w_end = (i_w + 1) * MAX_PATCH_SIZE
            print(f"process h_index: {i_h} w_index: {i_w}")
            with torch.no_grad():
                model_out = model(raw_image_in[:, :, h_start:h_end, w_start:w_end])
                # model_out = model(raw_image_in[:, :, 500:1000, 1000:1500])

            if 'raw' in args.out_type:
                denoised_rgb = process.raw2srgb(model_out, red_g, blue_g, ccm)
            elif 'linrgb' in args.out_type:
                denoised_rgb = process.rgb2srgb(model_out, red_g, blue_g, ccm)

            img_out[:, :, h_start*scale:h_end*scale, w_start*scale:w_end*scale] = denoised_rgb.detach().cpu()
            # print(f"processed h_index: {i_h} w_index: {i_w}")
    file_name = os.path.basename(args.test_data).split('.')[0]
    save_name = f'{file_name}-{os.path.basename(args.pretrain)}.png'
    save_image_tensor2cv2(img_out, os.path.join(args.save_dir, save_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args
    main()
