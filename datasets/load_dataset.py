import numpy as np
import torch
import torch.utils.data as data
import random
import h5py
from scipy.io import loadmat
from TorchTools.DataTools.Prepro import aug_img, aug_img_np, crop_img, crop_img_np, downsample_tensor, rggb_prepro
from model.common import DownsamplingShuffle
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from datasets import unprocess, process
from datasets.unprocess import mosaic
from tqdm import tqdm


class LoadPixelShiftData(data.Dataset):
    """
    load training simulated datasets
    """

    def __init__(self,
                 data_list,
                 phase='train',
                 patch_size=64,
                 downsampler='bic', scale=2,
                 in_type='noisy_lr_raw',
                 mid_type='raw',  # or None
                 out_type='rgb',
                 bit=14):
        super(LoadPixelShiftData, self).__init__()
        self.phase = phase
        self.scale = scale
        self.patch_size = patch_size
        self.downsampler = downsampler
        self.raw_stack = DownsamplingShuffle(2)
        self.in_type = in_type
        self.mid_type = mid_type
        self.out_type = out_type
        self.bit = bit

        # read image list from txt
        self.data_lists = []
        with open(data_list, 'r') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, desc='loading the {}'.format(data_list)):
                line = line.strip().split()
                self.data_lists.append(line[0])

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size

        # Note: RGGB is not RAW, it is a Full color sampled image.
        # (compose: Red, Green_red, Green_blue and blue channel);
        if self.phase == 'train':
            # for training dataset, the mat version if 4 (reads with scipy)
            matfile = loadmat(self.data_lists[index])
            rggb = np.asarray(matfile['raw']).astype(np.float32) / (2 ** self.bit - 1)
        else:
            # for testing dataset, the mat version if 7.3 (reads with h5py)
            with h5py.File(self.data_lists[index], 'r') as matfile:
                rggb = np.asarray(matfile['raw']).astype(np.float32) / (2 ** self.bit - 1)
                rggb = np.transpose(rggb, (2, 1, 0))
                matainfo = matfile['metadata']
                matainfo = {'colormatrix': np.transpose(matainfo['colormatrix']),
                            'red_gain': matainfo['red_gain'],
                            'blue_gain': matainfo['blue_gain']
                            }
                ccm, red_g, blue_g = process.metadata2tensor(matainfo)
                metadata = {'ccm': ccm, 'red_gain': red_g, 'blue_gain': blue_g}

        rggb = crop_img_np(rggb, self.patch_size, center_crop=self.phase != 'train')  # in PIL
        linrgb = np.stack((rggb[:, :, 0], np.mean(rggb[:, :, 1:3], axis=-1), rggb[:, :, 3]), axis=2)

        if self.phase == 'train':
            linrgb = aug_img_np(linrgb, random.randint(0, 7))
        linrgb = TF.to_tensor(linrgb)
        linrgb = torch.clamp(linrgb, 0., 1.)

        data = {'linrgb': linrgb.clone()}
        if 'lr' in self.in_type:
            lr_linrgb = downsample_tensor(linrgb, scale=self.scale, downsampler=self.downsampler)
            lr_linrgb = torch.clamp(lr_linrgb, 0., 1.)
            data.update({'lr_linrgb': lr_linrgb.clone()})

        # ---------------------------------
        # unprocess step
        # if raw, lin is in in_type, mean we need an unprocess
        if 'raw' in self.in_type:
            raw = unprocess.mosaic(linrgb)
            data.update({'raw': raw.clone()})

            if 'lr' in self.in_type:
                lr_raw = unprocess.mosaic(lr_linrgb)
                data.update({'lr_raw': lr_raw.clone()})

        if 'noisy' in self.in_type:
            shot_noise, read_noise = unprocess.random_noise_levels()
            if 'raw' in self.in_type:  # add noise to the bayer raw image and denoise it
                if 'lr' in self.in_type:
                    # Approximation of variance is calculated using noisy image (rather than clean
                    # image), since that is what will be avaiable during evaluation.
                    noisy_lr_raw = unprocess.add_noise(lr_raw, shot_noise, read_noise)
                    variance = shot_noise * noisy_lr_raw + read_noise
                    data.update({'noisy_lr_raw': noisy_lr_raw.clone(), 'variance': variance.clone()})
                else:
                    noisy_raw = unprocess.add_noise(raw, shot_noise, read_noise)
                    variance = shot_noise * noisy_raw + read_noise
                    data.update({'noisy_raw': noisy_raw.clone(), 'variance': variance.clone()})

            elif 'linrgb' in self.in_type:  # also add noise on raw but denoise on RGB.
                if 'lr' in self.in_type:
                    noisy_lr_linrgb = unprocess.add_noise(lr_linrgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_lr_linrgb + read_noise
                    data.update({'noisy_lr_linrgb': noisy_lr_linrgb.clone(), 'variance': variance.clone()})
                else:
                    noisy_linrgb = unprocess.add_noise(linrgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_linrgb + read_noise
                    data.update({'noisy_linrgb': noisy_linrgb.clone(), 'variance': variance.clone()})

        # Here return the data we need
        in_data = {self.in_type: data[self.in_type]}
        in_data.update({self.out_type: data[self.out_type]})
        if 'noisy' in self.in_type:
            in_data.update({'variance': data['variance']})
        if self.mid_type is not None:
            in_data.update({self.mid_type: data[self.mid_type]})
        if self.phase != 'train':
            in_data.update({'metadata': metadata})
        del data
        return in_data


class LoadSimData(data.Dataset):
    """
    load training simulated datasets
    """

    def __init__(self,
                 data_list,
                 phase='train',
                 patch_size=64,
                 downsampler='bic', scale=2,
                 in_type='noisy_lr_raw',
                 mid_type='raw',  # or None
                 out_type='rgb',
                 ):
        super(LoadSimData, self).__init__()
        self.phase = phase
        self.scale = scale
        self.patch_size = patch_size
        self.downsampler = downsampler
        self.raw_stack = DownsamplingShuffle(2)
        self.in_type = in_type
        self.mid_type = mid_type
        self.out_type = out_type

        # read image list from txt
        self.data_lists = []
        with open(data_list, 'r') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, desc='loading the {}'.format(data_list)):
                line = line.strip().split()
                self.data_lists.append(line[0])

        self.train_transforms = transforms.Compose([
            transforms.RandomCrop(self.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        self.test_transforms = transforms.Compose([
            transforms.CenterCrop(self.patch_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images
        rgb = Image.open(self.data_lists[index]).convert('RGB')
        if self.phase == 'train':
            rgb = self.train_transforms(rgb)
        else:
            rgb = self.test_transforms(rgb)

        data = {'rgb': rgb}
        if 'lr' in self.in_type:
            lr_rgb = downsample_tensor(rgb, scale=self.scale, downsampler=self.downsampler)
            lr_rgb = torch.clamp(lr_rgb, 0., 1.)
            data.update({'lr_rgb': lr_rgb.clone()})

        # ---------------------------------
        # unprocess step
        # if raw, lin is in in_type, mean we need an unprocess
        # if noisy is activated, we also need unprocess the rgb (since we only know the shot and read noise of the Raw).
        if 'raw' in self.in_type or 'lin' in self.in_type:
            raw = mosaic(rgb.clone())
            data.update({'raw': raw})

            if 'lr' in self.in_type:
                lr_raw = mosaic(lr_rgb.clone())
                data.update({'lr_raw': lr_raw.clone()})

        if 'noisy' in self.in_type:
            shot_noise, read_noise = unprocess.random_noise_levels()
            if 'raw' in self.in_type:  # add noise to the bayer raw image and denoise it
                if 'lr' in self.in_type:
                    # Approximation of variance is calculated using noisy image (rather than clean
                    # image), since that is what will be avaiable during evaluation.
                    noisy_lr_raw = unprocess.add_noise(lr_raw, shot_noise, read_noise)
                    variance = shot_noise * noisy_lr_raw + read_noise
                    data.update({'noisy_lr_raw': noisy_lr_raw.clone(), 'variance': variance.clone()})
                else:
                    noisy_raw = unprocess.add_noise(raw, shot_noise, read_noise)
                    variance = shot_noise * noisy_raw + read_noise
                    data.update({'noisy_raw': noisy_raw.clone(), 'variance': variance.clone()})

            elif 'rgb' in self.in_type:
                if 'lr' in self.in_type:
                    noisy_lr_rgb = unprocess.add_noise(lr_rgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_lr_rgb + read_noise
                    data.update({'noisy_lr_rgb': noisy_lr_rgb.clone(), 'variance': variance.clone()})
                else:
                    noisy_rgb = unprocess.add_noise(rgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_rgb + read_noise
                    data.update({'noisy_rgb': noisy_rgb.clone(), 'variance': variance.clone()})

        # Here return the data we need
        in_data = {self.in_type: data[self.in_type]}
        in_data.update({self.out_type: data[self.out_type]})
        if self.mid_type is not None:
            in_data.update({self.mid_type: data[self.mid_type]})
        if 'noisy' in self.in_type:
            in_data.update({'variance': variance})
        del data
        return in_data


class LoadSimDataUnproc(data.Dataset):
    """
    load training simulated datasets
    """

    def __init__(self,
                 data_list,
                 phase='train',
                 patch_size=64,
                 downsampler='bic', scale=2,
                 in_type='noisy_lr_raw',
                 mid_type='raw',  # or None
                 out_type='rgb',
                 ):
        super(LoadSimDataUnproc, self).__init__()
        self.phase = phase
        self.scale = scale
        self.patch_size = patch_size
        self.downsampler = downsampler
        self.raw_stack = DownsamplingShuffle(2)
        self.in_type = in_type
        self.mid_type = mid_type
        self.out_type = out_type

        # read image list from txt
        self.data_lists = []
        with open(data_list, 'r') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, desc='loading the {}'.format(data_list)):
                line = line.strip().split()
                self.data_lists.append(line[0])

        self.train_transforms = transforms.Compose([
            transforms.RandomCrop(self.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()
        ])
        self.test_transforms = transforms.Compose([
            transforms.CenterCrop(self.patch_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images
        srgb = Image.open(self.data_lists[index]).convert('RGB')
        if self.phase == 'train':
            srgb = self.train_transforms(srgb)
        else:
            srgb = self.test_transforms(srgb)

        data = {'srgb': srgb}
        if 'lr' in self.in_type:
            lr_srgb = downsample_tensor(srgb, scale=self.scale, downsampler=self.downsampler)
            lr_srgb = torch.clamp(lr_srgb, 0., 1.)
            data.update({'lr_srgb': lr_srgb.clone()})

        # ---------------------------------
        # unprocess step
        # if raw, lin is in in_type, mean we need an unprocess
        # if noisy is activated, we also need unprocess the rgb (since we only know the shot and read noise of the Raw).
        if 'raw' in self.in_type or 'lin' in self.in_type:
            rgb2cam = unprocess.random_ccm()
            cam2rgb = torch.inverse(rgb2cam)
            rgb_gain, red_gain, blue_gain = unprocess.random_gains()
            metadata = {
                'cam2rgb': cam2rgb,
                'rgb_gain': rgb_gain,
                'red_gain': red_gain,
                'blue_gain': blue_gain,
            }
            raw, linrgb = unprocess.unprocess(srgb, rgb2cam, rgb_gain, red_gain, blue_gain)

            data.update(metadata)
            data.update({'raw': raw.clone(), 'linrgb': linrgb.clone()})

            if 'lr' in self.in_type:
                lr_raw, lr_linrgb = unprocess.unprocess(lr_srgb, rgb2cam, rgb_gain, red_gain, blue_gain)
                data.update({'lr_raw': lr_raw.clone(), 'lr_linrgb': lr_linrgb.clone()})

        if 'noisy' in self.in_type:
            shot_noise, read_noise = unprocess.random_noise_levels()
            if 'raw' in self.in_type:  # add noise to the bayer raw image and denoise it
                if 'lr' in self.in_type:
                    # Approximation of variance is calculated using noisy image (rather than clean
                    # image), since that is what will be avaiable during evaluation.
                    noisy_lr_raw = unprocess.add_noise(lr_raw, shot_noise, read_noise)
                    variance = shot_noise * noisy_lr_raw + read_noise
                    data.update({'noisy_lr_raw': noisy_lr_raw.clone(), 'variance': variance.clone()})
                else:
                    noisy_raw = unprocess.add_noise(raw, shot_noise, read_noise)
                    variance = shot_noise * noisy_raw + read_noise
                    data.update({'noisy_raw': noisy_raw.clone(), 'variance': variance.clone()})

            elif 'linrgb' in self.in_type:  # also add noise on raw but denoise on RGB.
                if 'lr' in self.in_type:
                    noisy_lr_linrgb = unprocess.add_noise(lr_linrgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_lr_linrgb + read_noise
                    data.update({'noisy_lr_linrgb': noisy_lr_linrgb.clone(), 'variance': variance.clone()})
                else:
                    noisy_linrgb = unprocess.add_noise(linrgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_linrgb + read_noise
                    data.update({'noisy_linrgb': noisy_linrgb.clone(), 'variance': variance.clone()})
            elif 'srgb' in self.in_type:
                if 'lr' in self.in_type:
                    noisy_lr_srgb = unprocess.add_noise(lr_srgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_lr_srgb + read_noise
                    data.update({'noisy_lr_srgb': noisy_lr_srgb.clone(), 'variance': variance.clone()})
                else:
                    noisy_srgb = unprocess.add_noise(srgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_srgb + read_noise
                    data.update({'noisy_srgb': noisy_srgb.clone(), 'variance': variance.clone()})

        # Here return the data we need
        in_data = {self.in_type: data[self.in_type]}
        in_data.update({self.out_type: data[self.out_type]})
        if self.mid_type is not None:
            in_data.update({self.mid_type: data[self.mid_type]})
        if 'noisy' in self.in_type:
            in_data.update({'variance': data['variance']})
        in_data.update({'metadata': metadata})
        del data
        return in_data


class LoadBenchamrk(data.Dataset):
    """
    load training simulated datasets
    """

    def __init__(self,
                 data_list,
                 phase='train',
                 patch_size=64,
                 downsampler='bic', scale=2,
                 in_type='noisy_lr_raw',
                 mid_type='raw',  # or None
                 out_type='rgb',
                 ):
        super(LoadBenchamrk, self).__init__()
        self.phase = phase
        self.scale = scale
        self.patch_size = patch_size
        self.downsampler = downsampler
        self.raw_stack = DownsamplingShuffle(2)
        self.in_type = in_type
        self.mid_type = mid_type
        self.out_type = out_type

        # read image list from txt
        self.data_lists = []
        with open(data_list, 'r') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, desc='loading the {}'.format(data_list)):
                line = line.strip().split()
                self.data_lists.append(line[0])

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images
        srgb = Image.open(self.data_lists[index]).convert('RGB')
        srgb = crop_img(srgb, self.patch_size, center_crop=self.phase != 'train')  # in PIL
        if self.phase == 'train':
            srgb = aug_img(srgb)
        srgb = TF.to_tensor(srgb)

        data = {'srgb': srgb}
        if 'lr' in self.in_type:
            lr_srgb = downsample_tensor(srgb)
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
                'cam2rgb': cam2rgb,
                'rgb_gain': rgb_gain,
                'red_gain': red_gain,
                'blue_gain': blue_gain,
            }
            raw, linrgb = unprocess.unprocess(srgb, rgb2cam, rgb_gain, red_gain, blue_gain)

            data.update(metadata)
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
                    variance = shot_noise * noisy_lr_raw + read_noise
                    data.update({'noisy_lr_raw': noisy_lr_raw, 'variance': variance})
                else:
                    noisy_raw = unprocess.add_noise(raw, shot_noise, read_noise)
                    variance = shot_noise * noisy_raw + read_noise
                    data.update({'noisy_raw': noisy_raw, 'variance': variance})

            elif 'linrgb' in self.in_type:  # also add noise on raw but denoise on RGB.
                if 'lr' in self.in_type:
                    noisy_lr_linrgb = unprocess.add_noise(lr_linrgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_lr_linrgb + read_noise
                    data.update({'noisy_lr_linrgb': noisy_lr_linrgb, 'variance': variance})
                else:
                    noisy_linrgb = unprocess.add_noise(linrgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_linrgb + read_noise
                    data.update({'noisy_linrgb': noisy_linrgb, 'variance': variance})
            elif 'srgb' in self.in_type:
                if 'lr' in self.in_type:
                    noisy_lr_srgb = unprocess.add_noise(lr_srgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_lr_srgb + read_noise
                    data.update({'noisy_lr_srgb': noisy_lr_srgb, 'variance': variance})
                else:
                    noisy_srgb = unprocess.add_noise(srgb, shot_noise, read_noise)
                    variance = shot_noise * noisy_srgb + read_noise
                    data.update({'noisy_linrgb': noisy_srgb, 'variance': variance})

        # Here return the data we need
        in_data = {self.in_type: data[self.in_type]}
        in_data.update({self.out_type: data[self.out_type]})
        if self.mid_type is not None:
            in_data.update({self.mid_type: data[self.mid_type]})
        if 'noisy' in self.in_type:
            in_data.update({'variance': variance})
        # in_data.update({'metadata': metadata})
        del data
        return in_data


# # Code for debug the PixelShift dataset
# from datasets import process
# def vis_numpy(x):
#     import matplotlib.pyplot as plt
#     plt.imshow(x, cmap='gray')
#     plt.show()
#
#
# def vis_tensor(tensor):
#     import torch
#     from torchvision import utils
#     import matplotlib.pyplot as plt
#
#     grid = utils.make_grid(tensor)
#     ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     # vis_numpy(ndarr)
#     plt.imshow(ndarr, cmap='gray')
#     plt.show()
# def raw_unpack(input):
#     import torch.nn as nn
#     demo = nn.PixelShuffle(2)
#     return demo(input)
#
# # show the rgb img:
# # vis_gray(srgb)
# vis_gray(linrgb.permute(1,2,0))
# vis_gray(lr_linrgb.permute(1,2,0))

# # show the gt raw:
# vis_tensor(raw_unpack(raw.unsqueeze(0)))
#
# # show the noisy raw:
# vis_gray(raw_unpack(noisy_lr_raw.unsqueeze(0)).squeeze())
