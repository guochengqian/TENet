import numpy as np
import argparse
import os
import os.path as osp
import torch
import torch.utils.data as data
import glob
import random
from TorchTools.DataTools.Prepro import aug_img, aug_img_np, crop_img, crop_img_np, downsample_tensor, rggb_prepro
import torchvision.transforms.functional as TF
from PIL import Image
from datasets import unprocess, process
from tqdm import tqdm
import h5py
import cv2
from scipy.io import loadmat


EXT = ['png', 'PNG', 'tiff', 'tif', 'TIFF', 'JPG', 'jgp', 'bmp', 'BMP']


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LoadBenchmark(data.Dataset):
    """
    load training simulated datasets
    """

    def __init__(self,
                 data_path,
                 downsampler='bic', scale=2,
                 in_type='noisy_lr_raw',
                 mid_type='raw',  # or None
                 out_type='rgb',
                 src_dir='data/benchmark/urban100/HR',
                 ext='png'
                 ):
        super(LoadBenchmark, self).__init__()
        self.data_path = data_path
        self.scale = scale
        self.downsampler = downsampler

        self.in_type = in_type
        self.mid_type = mid_type
        self.out_type = out_type

        self.src_dir = src_dir
        self.ext = ext
        self.datas = []
        self.data_lists = []

        if self.data_path is None:
            file_name = '_'.join([osp.basename(osp.dirname(self.src_dir)), self.in_type,
                                  self.out_type, 'x' + str(self.scale)]) + '.pt'
            self.data_path = osp.join(osp.dirname(self.src_dir), file_name)
        if not osp.exists(self.data_path):
            print('===> {} does not exist, generate now'.format(self.data_path))
            self.generate_benchmark()
        else:
            self.datas = torch.load(self.data_path)

    def generate_benchmark(self):
        # set the seed for benchmarking
        set_seed(0)
        self.data_lists = sorted(
            glob.glob(osp.join(self.src_dir, '*' + self.ext))
        )

        for i in tqdm(range(len(self.data_lists))):
            self.datas.append(self.process_img(i))
        torch.save(self.datas, self.data_path)
        print('===> save benchmark dataset to {}'.format(self.data_path))

    def process_img(self, index):
        srgb = Image.open(self.data_lists[index]).convert('RGB')
        srgb = TF.to_tensor(srgb)
        C, H, W = srgb.shape
        if (H % (self.scale * 2) != 0) or (W % (self.scale * 2) != 0):
            H = H - H % (self.scale * 2)
            W = W - W % (self.scale * 2)
            srgb = srgb[:, :H, :W]
        data = {'srgb': srgb}

        if 'lr' in self.in_type:
            lr_srgb = downsample_tensor(srgb, scale=self.scale, downsampler=self.downsampler)
            lr_srgb = torch.clamp(lr_srgb, 0., 1.)
            data.update({'lr_srgb': lr_srgb})

        # ---------------------------------
        # unprocess step
        # if raw, lin is in in_type, mean we need an unprocess
        # if noisy is activated, we also need unprocess the rgb (since we only know the shot and read noise of the Raw).
        if 'raw' in self.in_type or 'lin' in self.in_type:
            rgb2cam = unprocess.random_ccm()
            cam2rgb = torch.inverse(rgb2cam)
            rgb_gain, red_gain, blue_gain = unprocess.random_gains()
            metadata = {
                'ccm': cam2rgb,
                'rgb_gain': rgb_gain,
                'red_gain': red_gain,
                'blue_gain': blue_gain,
            }
            raw, linrgb = unprocess.unprocess(srgb, rgb2cam, rgb_gain, red_gain, blue_gain)

            data.update({'metadata': metadata})
            data.update({'raw': raw, 'linrgb': linrgb})

            if 'lr' in self.in_type:
                lr_raw, lr_linrgb = unprocess.unprocess(lr_srgb, rgb2cam, rgb_gain, red_gain, blue_gain)
                data.update({'lr_raw': lr_raw, 'lr_linrgb': lr_linrgb})

        if 'noisy' in self.in_type:
            shot_noise, read_noise = unprocess.random_noise_levels()
            if 'raw' in self.in_type:  # add noise to the bayer raw image and denoise it
                if 'lr' in self.in_type:
                    # Approximation of variance is calculated using noisy image (rather than clean
                    # image), since that is what will be avaiable during evaluation.
                    noisy_lr_raw = unprocess.add_noise(lr_raw, shot_noise, read_noise)
                    data.update({'noisy_lr_raw': noisy_lr_raw})
                else:
                    noisy_raw = unprocess.add_noise(raw, shot_noise, read_noise)

                    data.update({'noisy_raw': noisy_raw})

            elif 'linrgb' in self.in_type:  # also add noise on raw but denoise on RGB.
                if 'lr' in self.in_type:
                    noisy_lr_linrgb = unprocess.add_noise(lr_linrgb, shot_noise, read_noise)
                    data.update({'noisy_lr_linrgb': noisy_lr_linrgb})
                else:
                    noisy_linrgb = unprocess.add_noise(linrgb, shot_noise, read_noise)
                    data.update({'noisy_linrgb': noisy_linrgb})
            elif 'srgb' in self.in_type:
                if 'lr' in self.in_type:
                    noisy_lr_srgb = unprocess.add_noise(lr_srgb, shot_noise, read_noise)
                    data.update({'noisy_lr_srgb': noisy_lr_srgb})
                else:
                    noisy_srgb = unprocess.add_noise(srgb, shot_noise, read_noise)
                    data.update({'noisy_linrgb': noisy_srgb})
            noise = {
                'read_noise': read_noise,
                'shot_noise': shot_noise,
            }
            data.update({'noise': noise})
        # Here return the data we need
        return data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # read images
        return self.datas[index]


class LoadBenchmarkPixelShift(data.Dataset):
    """
    load training simulated datasets
    """

    def __init__(self,
                 data_path,
                 downsampler='bic', scale=2,
                 in_type='noisy_lr_raw',
                 mid_type='raw',  # or None
                 out_type='rgb',
                 src_dir='data/benchmark/pixelshift200/gt',
                 ext='mat',
                 bit=14,
                 save_srgb=False,
                 ):
        super(LoadBenchmarkPixelShift, self).__init__()
        self.data_path = data_path
        self.scale = scale
        self.downsampler = downsampler

        self.in_type = in_type
        self.mid_type = mid_type
        self.out_type = out_type

        self.src_dir = src_dir
        self.ext = ext
        self.bit = bit
        self.save_srgb = save_srgb

        self.datas = []
        self.data_lists = sorted(glob.glob(osp.join(self.src_dir, '*' + self.ext)))

        if self.data_path is None:
            file_name = '_'.join([osp.basename(osp.dirname(self.src_dir)), self.in_type,
                                  self.out_type, 'x' + str(self.scale)]) + '.pt'
            self.data_path = osp.join(osp.dirname(self.src_dir), file_name)

        self.srgb_path = osp.join(osp.dirname(self.data_path), 'srgb')
        os.makedirs(self.srgb_path, exist_ok=True)

        if not osp.exists(self.data_path):
            print('===> {} does not exist, generate now'.format(self.data_path))
            self.generate_benchmark()
        else:
            self.datas = torch.load(self.data_path)

    def generate_benchmark(self):
        # set the seed for benchmarking
        set_seed(0)
        self.data_lists = sorted(
            glob.glob(osp.join(self.src_dir, '*' + self.ext))
        )

        for i in tqdm(range(len(self.data_lists))):
            self.datas.append(self.process_img(i))
        torch.save(self.datas, self.data_path)
        print('===> save benchmark dataset to {}'.format(self.data_path))

    def process_img(self, index):
        img_path = self.data_lists[index]
        data = {}

        try:
            with h5py.File(img_path, 'r') as matfile:
                # be carefull, save as mat higher version will transpose matrix.
                rggb = np.asarray(matfile['raw']).astype(np.float32) / (2 ** self.bit - 1)
                rggb = np.transpose(rggb, (2, 1, 0))
                matainfo = matfile['metadata']
                matainfo = {'colormatrix': np.transpose(matainfo['colormatrix']),
                            'red_gain': matainfo['red_gain'],
                            'blue_gain': matainfo['blue_gain']
                            }
                ccm, red_g, blue_g = process.metadata2tensor(matainfo)
                metadata = {'ccm': ccm, 'red_gain': red_g, 'blue_gain': blue_g}
                data.update({'metadata': metadata})

        except:
            matfile = loadmat(img_path)
            rggb = np.asarray(matfile['raw']).astype(np.float32) / (2 ** self.bit - 1)

        linrgb = np.stack((rggb[:, :, 0], np.mean(rggb[:, :, 1:3], axis=-1), rggb[:, :, 3]), axis=2)
        linrgb = TF.to_tensor(linrgb)
        linrgb = torch.clamp(linrgb, 0., 1.)
        data.update({'linrgb': linrgb})

        # get the srgb gt for pixelshift200 dataset.
        if self.save_srgb:
            srgb = process.rgb2srgb(linrgb.unsqueeze(0), metadata['red_gain'], metadata['blue_gain'], metadata['ccm']).squeeze(0)
            save_name = osp.basename(img_path).split('.')[0] + '.png'
            save_path = osp.join(self.srgb_path, save_name)
            ndarr = srgb.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            cv2.imwrite(save_path, cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if 'lr' in self.in_type:
            lr_linrgb = downsample_tensor(linrgb, scale=self.scale, downsampler=self.downsampler)
            lr_linrgb = torch.clamp(lr_linrgb, 0., 1.)
            data.update({'lr_linrgb': lr_linrgb})

        # ---------------------------------
        # unprocess step
        # if raw, lin is in in_type, mean we need an unprocess
        if 'raw' in self.in_type:
            raw = unprocess.mosaic(linrgb)
            data.update({'raw': raw})

            if 'lr' in self.in_type:
                lr_raw = unprocess.mosaic(lr_linrgb)
                data.update({'lr_raw': lr_raw})

        if 'noisy' in self.in_type:
            shot_noise, read_noise = unprocess.random_noise_levels()
            if 'raw' in self.in_type:  # add noise to the bayer raw image and denoise it
                if 'lr' in self.in_type:
                    # Approximation of variance is calculated using noisy image (rather than clean
                    # image), since that is what will be avaiable during evaluation.
                    noisy_lr_raw = unprocess.add_noise(lr_raw, shot_noise, read_noise)
                    data.update({'noisy_lr_raw': noisy_lr_raw})
                else:
                    noisy_raw = unprocess.add_noise(raw, shot_noise, read_noise)
                    data.update({'noisy_raw': noisy_raw})

            elif 'linrgb' in self.in_type:  # also add noise on raw but denoise on RGB.
                if 'lr' in self.in_type:
                    noisy_lr_linrgb = unprocess.add_noise(lr_linrgb, shot_noise, read_noise)
                    data.update({'noisy_lr_linrgb': noisy_lr_linrgb})
                else:
                    noisy_linrgb = unprocess.add_noise(linrgb, shot_noise, read_noise)
                    data.update({'noisy_linrgb': noisy_linrgb})
            noise = {
                'read_noise': read_noise,
                'shot_noise': shot_noise,
            }
            data.update({'noise': noise})
        return data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # read images
        return self.datas[index]


if __name__ == '__main__':
    """
    Given a batch of images (RGB in PNG or RGGB in MAT), 
    load all of them into .pth file, and use them as evaluation datasets. 
    """

    parser = argparse.ArgumentParser(description='Evaluation Data preparation')
    parser.add_argument('--in_type', type=str, default='noisy_lr_raw')
    parser.add_argument('--mid_type', type=str, default='raw')
    parser.add_argument('--out_type', type=str, default='linrgb')

    parser.add_argument('--ext', type=str, default='png')

    parser.add_argument('--scale', type=int, default=2,
                        help='default scale ratio for the SR')
    parser.add_argument('--save_srgb', action='store_true',
                        help='save srgb images of the desired dataset for preview')
    parser.add_argument('--src_dir',
                        default='data/benchmark/pixelshift200/gt',
                        help='path to the original data')

    args = parser.parse_args()

    project_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))

    src_dir = osp.join(project_dir, args.src_dir)

    if 'pixelshift' in args.src_dir.lower():
        benchmark_data = LoadBenchmarkPixelShift(None,
                                                 in_type=args.in_type,
                                                 mid_type=args.mid_type,
                                                 out_type=args.out_type,
                                                 scale=args.scale,
                                                 src_dir=src_dir,
                                                 save_srgb=args.save_srgb,
                                                 )
    else:
        benchmark_data = LoadBenchmark(None,
                                       in_type=args.in_type,
                                       mid_type=args.mid_type,
                                       out_type=args.out_type,
                                       scale=args.scale,
                                       src_dir=src_dir,
                                       ext=args.ext)

# # Code for debug the PixelShift dataset
# from datasets import process
# def vis_numpy(x):
#     import matplotlib.pyplot as plt
#     plt.imshow(x, cmap='gray')
#     plt.show()
#
#
# def vis_tensor(tensor):
#     from torchvision import utils
#     import matplotlib.pyplot as plt
#
#     grid = utils.make_grid(tensor)
#     ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     # vis_numpy(ndarr)
#     plt.imshow(ndarr, cmap='gray')
#     plt.show()
#
#
#
# def raw_unpack(input):
#     import torch.nn as nn
#     demo = nn.PixelShuffle(2)
#     return demo(input)

# # show the rgb img:
# # vis_gray(srgb)
# vis_gray(linrgb.permute(1,2,0))
# vis_gray(lr_linrgb.permute(1,2,0))
#
# # show the gt raw:
# vis_tensor(raw_unpack(raw.unsqueeze(0)))
#
# # show the noisy raw:
# vis_gray(raw_unpack(noisy_lr_raw.unsqueeze(0)).squeeze())
