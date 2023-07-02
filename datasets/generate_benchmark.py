import numpy as np
import argparse
import os
import os.path as osp
import torch
import torch.utils.data as data
import glob
import random
from TorchTools.DataTools.Prepro import aug_img, aug_img_np, crop_img, crop_img_np, downsample_tensor, rggb_prepro
from TorchTools.DataTools.FileTools import tensor2np
import torchvision.transforms.functional as TF
from PIL import Image
from datasets import unprocess, process
from datasets.unprocess import mosaic
from tqdm import tqdm
import h5py
import cv2
from scipy.io import loadmat, savemat


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
    Load RGB GT images -> generate input 
    """
    def __init__(self,
                 data_dir='data/benchmark',
                 downsampler='bic', scale=2,
                 in_type='noisy_lr_raw',
                 mid_type='raw',  # or None
                 out_type='rgb',
                 ext='png', 
                 noise_model='gp', 
                 sigma=10
                 ):
        super(LoadBenchmark, self).__init__()
        self.data_dir = data_dir
        self.scale = scale
        self.downsampler = downsampler
        self.noise_model = noise_model
        self.sigma = sigma / 255.

        self.in_type = in_type
        self.mid_type = mid_type
        self.out_type = out_type
        
        filename = f'x{scale}_{noise_model}' if 'p' in noise_model else f'x{scale}_{noise_model}x{sigma}' 
        self.save_dir = osp.join(data_dir, filename)
        self.src_dir = osp.join(data_dir, 'gt')
        self.data_path = osp.join(data_dir, f'{filename}.pth')
        os.makedirs(self.save_dir, exist_ok=True)

        self.ext = ext
        self.datas = []
        self.data_lists = []
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
        rgb = Image.open(self.data_lists[index]).convert('RGB')
        rgb = TF.to_tensor(rgb)
        C, H, W = rgb.shape
        if (H % (self.scale * 2) != 0) or (W % (self.scale * 2) != 0):
            H = H - H % (self.scale * 2)
            W = W - W % (self.scale * 2)
            rgb = rgb[:, :H, :W]
        data = {'rgb': rgb}

        if 'lr' in self.in_type:
            lr_rgb = downsample_tensor(rgb, scale=self.scale, downsampler=self.downsampler)
            lr_rgb = torch.clamp(lr_rgb, 0., 1.)
            data.update({'lr_rgb': lr_rgb})

        # ---------------------------------
        # unprocess step
        # if raw, lin is in in_type, mean we need an unprocess
        # if noisy is activated, we also need unprocess the rgb (since we only know the shot and read noise of the Raw).
        if  ('raw' in self.in_type and 'p' in self.noise_model) or 'lin' in self.in_type:
            rgb2cam = unprocess.random_ccm()
            cam2rgb = torch.inverse(rgb2cam)
            rgb_gain, red_gain, blue_gain = unprocess.random_gains()
            metadata = {
                'ccm': cam2rgb,
                'rgb_gain': rgb_gain,
                'red_gain': red_gain,
                'blue_gain': blue_gain,
            }
            raw, linrgb = unprocess.unprocess(rgb, rgb2cam, rgb_gain, red_gain, blue_gain)
            data.update({'metadata': metadata})
            data.update({'raw': raw, 'linrgb': linrgb})
            if 'lr' in self.in_type:
                lr_raw, lr_linrgb = unprocess.unprocess(lr_rgb, rgb2cam, rgb_gain, red_gain, blue_gain)
                data.update({'lr_raw': lr_raw, 'lr_linrgb': lr_linrgb})

        if  ('raw' in self.in_type and 'p' not in self.noise_model) or 'lin' in self.in_type:
            raw = mosaic(rgb.clone())
            data.update({'raw': raw})
            if 'lr' in self.in_type:
                lr_raw = mosaic(lr_rgb.clone())
                data.update({'lr_raw': lr_raw})

        if 'noisy' in self.in_type and 'p' in self.noise_model:
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
            elif 'rgb' in self.in_type:
                if 'lr' in self.in_type:
                    noisy_lr_rgb = unprocess.add_noise(lr_rgb, shot_noise, read_noise)
                    data.update({'noisy_lr_rgb': noisy_lr_rgb})
                else:
                    noisy_rgb = unprocess.add_noise(rgb, shot_noise, read_noise)
                    data.update({'noisy_linrgb': noisy_rgb})
            noise = {
                'read_noise': read_noise,
                'shot_noise': shot_noise,
            }
            data.update({'noise': noise})

        if 'noisy' in self.in_type and 'p' not in self.noise_model:
            if 'raw' in self.in_type:  # add noise to the bayer raw image and denoise it
                if 'lr' in self.in_type:
                    # Approximation of variance is calculated using noisy image (rather than clean
                    # image), since that is what will be avaiable during evaluation.
                    noisy_lr_raw = lr_raw + torch.randn(lr_raw.size()).mul_(self.sigma)
                    variance = torch.ones(lr_raw[0:1].size()).mul_(self.sigma)
                    data.update({'noisy_lr_raw': noisy_lr_raw.clone(), 'variance': variance.clone()})
                else:
                    noisy_raw = raw + torch.randn(raw.size()).mul_(self.sigma)
                    variance = torch.ones(raw[0:1].size()).mul_(self.sigma) 
                    data.update({'noisy_raw': noisy_raw.clone(), 'variance': variance.clone()})
            elif 'rgb' in self.in_type:
                if 'lr' in self.in_type:
                    noisy_lr_rgb = lr_rgb + torch.randn(lr_rgb.size()).mul_(self.sigma)
                    variance = torch.ones(lr_rgb[0:1].size()).mul_(self.sigma) 
                    data.update({'noisy_lr_rgb': noisy_lr_rgb.clone(), 'variance': variance.clone()})
                else:
                    noisy_rgb = rgb + torch.randn(rgb.size()).mul_(self.sigma)
                    variance = torch.ones(rgb[0:1].size()).mul_(self.sigma) 
                    data.update({'noisy_rgb': noisy_rgb.clone(), 'variance': variance.clone()})
        
        # save for matlab
        if 'noisy_lr_raw' in data.keys():
            filename = osp.basename(self.data_lists[index].split('.')[0] + '.mat') 
            savemat(osp.join(self.save_dir, filename), {'noisy_lr_raw': data['noisy_lr_raw'].permute(1, 2, 0).numpy()})
        return data

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        # read images
        return self.datas[index]


class LoadBenchmarkPixelShift(data.Dataset):
    def __init__(self,
                 data_path,
                 downsampler='bic', scale=2,
                 in_type='noisy_lr_raw',
                 mid_type='raw',  # or None
                 out_type='linrgb',
                 src_dir='data/benchmark/pixelshift/mat',
                 ext='mat',
                 bit=14,
                 ):
        super(LoadBenchmarkPixelShift, self).__init__()
        self.data_path = osp.join(data_path, f'pixelshift_{in_type}_{out_type}_x{scale}.pt')
        self.scale = scale
        self.downsampler = downsampler
        self.in_type = in_type
        self.mid_type = mid_type
        self.out_type = out_type

        self.src_dir = src_dir
        self.gt_dir = osp.join(osp.dirname(src_dir), 'gt')
        self.ext = ext
        self.bit = bit

        self.datas = []
        self.data_lists = sorted(glob.glob(osp.join(self.src_dir, '*')))

        if not osp.exists(self.data_path):
            print('===> {} does not exist, generate now'.format(self.data_path))
            self.generate_benchmark()
        else:
            self.datas = torch.load(self.data_path)

    def generate_benchmark(self):
        # set the seed for benchmarking
        set_seed(0)
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

        # get the rgb gt for pixelshift200 dataset.
        rgb = process.rgb2srgb(linrgb.unsqueeze(0), metadata['red_gain'], metadata['blue_gain'], metadata['ccm']).squeeze(0)
        save_name = osp.basename(img_path).split('.')[0] + '.png'
        save_path = osp.join(self.gt_dir, save_name)
        ndarr = rgb.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        cv2.imwrite(save_path, cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        data.update({'rgb': rgb})
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
    parser.add_argument('--save_rgb', action='store_true',
                        help='save rgb images of the desired dataset for preview')
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
                                                 save_rgb=args.save_rgb,
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
# # vis_gray(rgb)
# vis_gray(linrgb.permute(1,2,0))
# vis_gray(lr_linrgb.permute(1,2,0))
#
# # show the gt raw:
# vis_tensor(raw_unpack(raw.unsqueeze(0)))
#
# # show the noisy raw:
# vis_gray(raw_unpack(noisy_lr_raw.unsqueeze(0)).squeeze())
