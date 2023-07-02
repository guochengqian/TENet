import cv2
import argparse
import os
from TorchTools.DataTools.FileTools import _all_images, _tensor2cvimage
from TorchTools.DataTools.Prepro import rgb2raw, add_noise
from tqdm import tqdm
import torch
import numpy as np
from model.common import DownsamplingShuffle

'''
src_path: folder containing RGB images, rgb_img_1.png  (or jpg);
dst_path: folder to put raw images, save as rgb_img_1.tiff
bayer_pattern: bayer_pattern, default is rggb
'''
# ===================> load parameters
parser = argparse.ArgumentParser(description='RGB2RAW implementation')
parser.add_argument('--src_path', type=str, default='/data/image/TENet/TENet_test_data/TENet_sim_test/gt/',
                    help='folder containing RGB images, rgb_img_1.png  (or jpg)')
parser.add_argument('--dst_path', type=str, default='/data/image/TENet/TENet_test_data/TENet_sim_test/noisy_input/x2/sigma10',
                    help='folder to put raw images, save as rgb_img_1.tiff')
parser.add_argument('--bayer', type=str, default='rggb', help='bayer_pattern, default is rggb')
parser.add_argument('--scale', type=int, default=2, help='downsample raw?')
parser.add_argument('--shot_noise', type=float, default=0, help='add shot noise (0,255)')
parser.add_argument('--read_noise', type=float, default=10, help='add read noise (0,255)')
parser.add_argument('--n_folders', type=int, default=2, help='number of images folders')

opt = parser.parse_args()

src_path = opt.src_path
dst_path = opt.dst_path

print('==================> load src_images')
im_files = _all_images(src_path)
for index in tqdm(range(len(im_files))):
    im_file = im_files[index]
    save_name = os.path.basename(im_file).split('.')[0]+'.tiff'

    if opt.n_folders > 1:
        sub_folder = im_file.split('/')[-2]
        dst_path = os.path.join(opt.dst_path, sub_folder)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)

    # read image and reshape
    rgb = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)
    h, w, c = rgb.shape
    h = h - h % (2 * opt.scale)
    w = w - w % (2 * opt.scale)
    rgb = rgb[0:h, 0:w]
    rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))).float()

    # down-sampling
    if opt.scale > 1:
        rgb = torch.nn.functional.avg_pool2d(rgb, opt.scale, opt.scale)
    raw = rgb2raw(rgb, True, opt.bayer).unsqueeze_(0)

    # add noise
    if opt.shot_noise>0 or opt.read_noise>0:
        raw = add_noise(raw, opt.shot_noise, opt.read_noise)
    raw = torch.clamp(raw, 0, 255.)[0, 0].numpy().astype(np.uint8)
    cv2.imwrite(os.path.join(dst_path, save_name), raw)

