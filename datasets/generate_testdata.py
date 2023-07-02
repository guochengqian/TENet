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
import h5py
from glob import glob
from model.common import raw_unpack, raw_pack
from TorchTools.DataTools.FileTools import save_tensor_to_cv2img


color1 = [[1.087177634, -0.4568284452, -0.1467799544], [-0.4628916085, 1.416070819, 0.2814568579],
          [-0.02583973669, 0.0994599387, 0.7301111817]]


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
        # xyz2cam = raw_file.rgb_xyz_matrix[0:3, 0:3]
        colormatrix = color1

        # colormatrix = (raw_file.color_matrix[0:3, 0:3])
        gains = raw_file.daylight_whitebalance
        rgb_gain = [gains[1]]
        red_gain = [gains[0] / gains[1]]
        blue_gain = [gains[2] / gains[1]]

        metadata = {
            # 'colormatrix': xyz2cam,
            'colormatrix': colormatrix,
            'red_gain': red_gain,
            'blue_gain': blue_gain,
        }

    if debug:
        ccm, red_g, blue_g = process.metadata2tensor(metadata)

        # Below is for testing
        raw = torch.unsqueeze(TF.to_tensor(bayer_image), dim=0)
        raw = raw_pack(raw)

        srgb = process.raw2srgb(raw, red_g, blue_g, ccm)

        # here, jpg is only for debug.
        unpack_raw = raw_unpack(raw)
        utils.save_image(unpack_raw, osp.join(save_path, src_name + '_raw.jpg'))
        utils.save_image(srgb, osp.join(save_path, src_name + 'rgb.jpg'))

        utils.save_image(srgb[:, :, 1000:1200, 2800:3000], osp.join(save_path, src_name + 'rgbcropped.jpg'))

    return bayer_image, metadata


def save_image_raw(src_path,
                   save_dir,
                   img_save_dir,
                   debug=False):
    file_name = osp.basename(src_path)
    root_name, ext = file_name.split('.')[:]
    save_path = osp.join(save_dir, root_name + '.pth')

    bayer_image, metadata = process_single_raw(src_path, save_dir, debug=debug)
    # bayer_image = bayer_image[2:-2,2:-2]

    height, width = bayer_image.shape
    bayer_images = []
    for yy in range(2):
        for xx in range(2):
            image_c = bayer_image[yy:height:2, xx:width:2].copy()
            bayer_images.append(image_c)

    linrggb = np.stack((bayer_images[0], bayer_images[1], bayer_images[2], bayer_images[3]), axis=2)
    save_mat = {'rggb': np.floor(linrggb * (2 ** BIT - 1)).astype(np.int16), 'metadata': metadata}
    torch.save(save_mat, save_path)

    linrgb = np.stack((linrggb[:, :, 0], (linrggb[:, :, 1] / 2 + linrggb[:, :, 2] / 2), linrggb[:, :, 3]), axis=2)
    linrgb = torch.unsqueeze(TF.to_tensor(linrgb.astype(np.float32)), dim=0)
    # utils.save_image(linrgb, osp.join(img_save_path, root_name + 'linrgb.jpg'))

    ccm, red_g, blue_g = process.metadata2tensor(metadata)
    srgb = process.rgb2srgb(linrgb, red_g, blue_g, ccm)
    save_tensor_to_cv2img(srgb, osp.join(img_save_dir, root_name + 'srgb.jpg'))


def main(src_dir,
         save_dir,
         img_save_dir,
         ext='dng'):
    src_list = glob(osp.join(src_dir, "*." + ext))
    src_list.sort()

    for i, raw_path in enumerate(src_list):
        save_image_raw(raw_path, save_dir, img_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    parser.add_argument('--data_dir', type=str, default='/data/lowlevel/ISPNet/orgdata',
                        help='path to original ARW')

    parser.add_argument('--ext', type=str, default='dng',
                        help='type of raw image. ARW? DNG? dng?')
    parser.add_argument('--bit', type=str, default=12,
                        help='bit of pixel intensity')
    args = parser.parse_args()
    BIT = args.bit
    EXT = args.ext

    # assume raw images (in ARW format for Sony) are in $data_dir/original/ARW
    raw_dir = osp.join(args.data_dir, 'original/DNG')

    # will save final rggb images in $data_dir/mat
    save_dir = osp.join(args.data_dir, 'mat')

    # will save matadata in $data_dir
    # save preview images in $data_dir/img
    preview_img_dir = osp.join(args.data_dir, 'preview')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(preview_img_dir, exist_ok=True)
    main(raw_dir, save_dir, preview_img_dir, ext=EXT)
