import os
import os.path as osp
import argparse
import numpy as np
import torch
import rawpy
import rawpy.enhance
from torchvision import utils
import torchvision.transforms.functional as TF
from datasets import process
import scipy.io as sio
from glob import glob
from model.common import raw_unpack, raw_pack

SHIFTPATTERN = [
    [0, 0],
    [-1, 0],
    [-1, 1],
    [0, 1]
]
EXIT = "ARW"
BIT = 14


def process_single_raw(src_path,
                       save_path,
                       debug=False,
                       metadata=None,
                       ):
    src_name = osp.basename(src_path).split('.')[0]

    # src_path
    raw_file = rawpy.imread(src_path)
    black_level = np.mean(raw_file.black_level_per_channel)
    white_level = float(raw_file.white_level)

    # raw image
    bayer_image = raw_file.raw_image.astype(np.float)
    bayer_image = (bayer_image - black_level) / (white_level - black_level)
    bayer_image = np.clip(bayer_image, 0, 1.0)

    # metadata
    if metadata is None:
        xyz2cam = raw_file.rgb_xyz_matrix[0:3, 0:3]

        gains = raw_file.daylight_whitebalance
        rgb_gain = [gains[1]]
        red_gain = [gains[0]/gains[1]]
        blue_gain = [gains[2]/gains[1]]

        metadata = {
            'colormatrix': xyz2cam,
            'rgb_gain': rgb_gain,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
        }

    if debug:
        ccm, red_g, blue_g = process.metadata2tensor(metadata)

        # Below is for testing
        raw = torch.unsqueeze(TF.to_tensor(bayer_image), dim=0)
        raw = raw_pack(raw)

        srgb = process.raw2srgb(raw, red_g,  blue_g, ccm)

        # here, jpg is only for debug.
        unpack_raw = raw_unpack(raw)
        utils.save_image(unpack_raw, osp.join(save_path, src_name+'_raw.jpg'))
        utils.save_image(srgb, osp.join(save_path, src_name+'rgb.jpg'))

        utils.save_image(srgb[:, :, 1000:1200, 2800:3000], osp.join(save_path, src_name+'rgbcropped.jpg'))

    return bayer_image, metadata


def process_4_pixel_shift_raws(src_path,
                               save_path,
                               img_save_path,
                               debug=False):
    file_dir = osp.dirname(src_path)
    file_name = osp.basename(src_path)
    root_name, ext = file_name.split('.')[:]
    root_name = root_name[:-1]
    save_file = osp.join(save_path, root_name + 'rggb.mat')

    bayer_images = []
    for i in range(4):
        raw_file_name = root_name+str(i+1) + '.' + ext
        raw_file_path = osp.join(file_dir, raw_file_name)

        if i == 0:
            bayer_image, metadata = process_single_raw(raw_file_path, save_path, debug=debug)
            bayer_image = bayer_image[2:-2, 2:-2]
        else:
            bayer_image, _ = process_single_raw(raw_file_path, save_path, metadata=metadata)
            bayer_image = bayer_image[2+SHIFTPATTERN[i][0]:-2+SHIFTPATTERN[i][0],
                                      2+SHIFTPATTERN[i][1]:-2+SHIFTPATTERN[i][1]]
        bayer_images.append(bayer_image)    # RGBG.

    rggb_order = [0, 3, 1, 2]
    bayer_images = [bayer_images[i] for i in rggb_order]

    # Begin the pixel shift color assignment
    # red channel
    red_channel = bayer_images[0].copy()
    red_channel[::2, 1::2] = bayer_images[1][::2, 1::2]
    red_channel[1::2, ::2] = bayer_images[2][1::2, ::2]
    red_channel[1::2, 1::2] = bayer_images[3][1::2, 1::2]

    # Gr channel
    green_r = bayer_images[1].copy()
    green_r[::2, 1::2] = bayer_images[0][::2, 1::2]
    green_r[1::2, ::2] = bayer_images[3][1::2, ::2]
    green_r[1::2, 1::2] = bayer_images[2][1::2, 1::2]

    # Gb channel
    green_b = bayer_images[2].copy()
    green_b[::2, 1::2] = bayer_images[3][0::2, 1::2]
    green_b[1::2, ::2] = bayer_images[0][1::2, ::2]
    green_b[1::2, 1::2] = bayer_images[1][1::2, 1::2]

    # red channel
    blue = bayer_images[3].copy()
    blue[::2, 1::2] = bayer_images[2][0::2, 1::2]
    blue[1::2, ::2] = bayer_images[1][1::2, ::2]
    blue[1::2, 1::2] = bayer_images[0][1::2, 1::2]

    linrggb = np.stack((red_channel, green_r, green_b, blue), axis=2)
    save_mat = {
        'rggb': np.floor(linrggb * (2**BIT-1)).astype(np.int16),
    }

    sio.savemat(save_file, save_mat)

    linrgb = np.stack((linrggb[:, :, 0], (linrggb[:, :, 1]/2+linrggb[:, :, 2]/2), linrggb[:, :, 3]), axis=2)
    linrgb = torch.unsqueeze(TF.to_tensor(linrgb), dim=0)
    # utils.save_image(linrgb, osp.join(img_save_path, root_name + 'linrgb.jpg'))

    ccm, red_g, blue_g = process.metadata2tensor(metadata)
    srgb = process.rgb2srgb(linrgb, red_g, blue_g, ccm)
    utils.save_image(srgb, osp.join(img_save_path, root_name + 'srgb.jpg'))
    return {root_name[:-1]: metadata}


def main(src_path,
         save_path,
         meta_save_path,
         img_save_path,
         ext="ARW"):
    src_list = glob(osp.join(src_path, "*"+ext))
    src_list.sort()

    metadatas= {}
    for i, raw_path in enumerate(src_list):
        if i % 4 == 0:
            metadata = process_4_pixel_shift_raws(raw_path, save_path, img_save_path)
            metadatas.update(metadata)
    sio.savemat(osp.join(meta_save_path, 'metadata.mat'), {'metadatas': metadatas})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    parser.add_argument('--data_dir', type=str, default='/data/lowlevel/ISPNet/pixelshift200',
                        help='path to original ARW')
    parser.add_argument('--debug', action="store_true",
                        help='enable debug mode')
    args = parser.parse_args()

    # assume raw images (in ARW format for Sony) are in $data_dir/original/ARW
    raw_dir = osp.join(args.data_dir, 'original/ARW')

    # will save final rggb images in $data_dir/mat
    save_dir = osp.join(args.data_dir, 'mat')

    # will save matadata in $data_dir
    # save preview images in $data_dir/img
    preview_img_dir = osp.join(args.data_dir, 'preview')

    # save debug images in $data_dir/debug folder
    debug_dir = osp.join(args.data_dir, 'debug')
    
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)
        _ = process_4_pixel_shift_raws(osp.join(raw_dir, "pixelshift_10_1.ARW"),
                                       debug_dir, debug_dir, debug=True)
    else:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(preview_img_dir, exist_ok=True)
        main(raw_dir, save_dir, args.data_dir, preview_img_dir)

