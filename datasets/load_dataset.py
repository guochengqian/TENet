import numpy as np
import torch
import torch.utils.data as data
import random
from scipy.io import loadmat
from TorchTools.DataTools.FileTools import _read_image
from TorchTools.DataTools.Prepro import rgb2raw, data_aug, rggb_prepro
    # add_noise, rgb_avg_downsample, rggb_prepro, data_aug
from model.common import DownsamplingShuffle
from scipy.misc import imresize
import torch.nn.functional as F
import cv2

class LoadPixelShiftData(data.Dataset):
    """
    load training simulated datasets

    """
    def __init__(self, data_list, patch_size=64, scale=2, denoise=False, max_noise=0.0748, min_noise=0.0,
                 downsampler='avg', get2label=False):
        super(LoadPixelShiftData, self).__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.data_lists = []
        self.denoise = denoise
        self.max_noise = max_noise
        self.min_noise = min_noise
        self.downsampler = downsampler
        self.raw_stack = DownsamplingShuffle(2)
        self.get2label = get2label
        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        rggb = np.asarray(loadmat(self.data_lists[index])['ps']).astype(np.float32)/32767.
        rggb = data_aug(rggb, mode=np.random.randint(0, 8))
        h, w, c = rggb.shape

        lr_raw, raw, rgb = rggb_prepro(rggb.copy(), self.scale)

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        rgb = rgb[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        lr_raw = lr_raw[:, hi//self.scale: hi//self.scale + self.patch_size//self.scale,
                 wi//self.scale: wi//self.scale + self.patch_size//self.scale]
        lr_raw = lr_raw.view(1, 1, lr_raw.shape[-2], lr_raw.shape[-1])
        lr_raw = self.raw_stack(lr_raw)

        if self.denoise:
            noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]

            # raw_input + noise
            noise = torch.randn([1, 1, self.patch_size//self.scale, self.patch_size//self.scale]).mul_(noise_level)
            lr_raw = lr_raw + self.raw_stack(noise)

            # cat noise_map
            noise_map = torch.ones([1, 1, self.patch_size//(2*self.scale), self.patch_size//(2*self.scale)])*noise_level
            lr_raw = torch.cat((lr_raw, noise_map), 1)


        data = {}
        data['input'] = torch.clamp(lr_raw, 0., 1.)[0]
        data['gt'] = torch.clamp(rgb, 0., 1.)
        if self.get2label:
            raw = raw[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
            data['raw_gt'] = torch.clamp(raw, 0., 1.)

        return data


class LoadSimData(data.Dataset):
    """
    load training simulated datasets

    """
    def __init__(self, data_list, patch_size=64, scale=2, denoise=False, max_noise=0.0748, min_noise=0.0,
                 downsampler='avg', get2label=False):
        super(LoadSimData, self).__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.data_lists = []
        self.denoise = denoise
        self.max_noise = max_noise
        self.min_noise = min_noise
        self.downsampler = downsampler
        self.raw_stack = DownsamplingShuffle(2)
        self.get2label = get2label
        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        rgb = cv2.cvtColor(cv2.imread(self.data_lists[index]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = data_aug(rgb, mode=np.random.randint(0, 8))
        h, w, c = rgb.shape
        h = h // 8 * 8
        w = w // 8 * 8
        rgb = rgb[0:h, 0:w, :]

        if self.downsampler == 'bic':
            lr_rgb = cv2.resize(rgb.copy(), (0,0), fx=1/self.scale, fy=1/self.scale, interpolation=cv2.INTER_CUBIC)
            lr_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(lr_rgb, [2, 0, 1]))).float()

        rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))).float()

        if self.downsampler == 'avg':
            lr_rgb = F.avg_pool2d(rgb.clone(), self.scale, self.scale)

        lr_raw = rgb2raw(lr_rgb, is_tensor=True)

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        rgb = rgb[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        lr_raw = lr_raw[:, hi//self.scale: hi//self.scale + self.patch_size//self.scale,
                 wi//self.scale: wi//self.scale + self.patch_size//self.scale]
        lr_raw = lr_raw.view(1, 1, lr_raw.shape[-2], lr_raw.shape[-1])
        lr_raw = self.raw_stack(lr_raw)

        if self.denoise:
            noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]

            # raw_input + noise
            noise = torch.randn([1, 1, self.patch_size//self.scale, self.patch_size//self.scale]).mul_(noise_level)
            lr_raw = lr_raw + self.raw_stack(noise)

            # cat noise_map
            noise_map = torch.ones([1, 1, self.patch_size//(2*self.scale), self.patch_size//(2*self.scale)])*noise_level
            lr_raw = torch.cat((lr_raw, noise_map), 1)


        data = {}
        data['input'] = torch.clamp(lr_raw, 0., 1.)[0]
        data['gt'] = torch.clamp(rgb, 0., 1.)
        if self.get2label:
            data['raw_gt'] = torch.clamp(rgb2raw(rgb.clone(), is_tensor=True), 0., 1.)

        return data

####################################################
# load datasets for abaltaion study
####################################################
class LoadDemo(data.Dataset):
    """
    load training simulated datasets

    """
    def __init__(self, data_list, patch_size=64, denoise=False, max_noise=0.0748, min_noise=0.0):
        super(LoadDemo, self).__init__()
        self.patch_size = patch_size
        self.data_lists = []
        self.denoise = denoise
        self.max_noise = max_noise
        self.min_noise = min_noise
        self.raw_stack = DownsamplingShuffle(2)
        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        rgb = cv2.cvtColor(cv2.imread(self.data_lists[index]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = data_aug(rgb, mode=np.random.randint(0, 8))
        h, w, c = rgb.shape
        h = h // 2 * 2
        w = w // 2 * 2
        rgb = rgb[0:h, 0:w, :]

        rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))).float()

        raw = rgb2raw(rgb.clone(), is_tensor=True)

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        rgb = rgb[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        raw = raw[:, hi: hi + self.patch_size, wi: wi + self.patch_size]

        raw = raw.view(1, 1, raw.shape[-2], raw.shape[-1])
        raw = self.raw_stack(raw)

        if self.denoise:
            noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]

            # raw_input + noise
            noise = torch.randn([1, 1, self.patch_size, self.patch_size]).mul_(noise_level)
            raw = raw + self.raw_stack(noise)

            # cat noise_map
            noise_map = torch.ones([1, 1, self.patch_size//2, self.patch_size//2])*noise_level
            raw = torch.cat((raw, noise_map), 1)


        data = {}
        data['input'] = torch.clamp(raw[0], 0., 1.)
        data['gt'] = torch.clamp(rgb, 0., 1.)

        return data


class LoadRawDeno(data.Dataset):
    """
    load training simulated datasets

    """
    def __init__(self, data_list, patch_size=64, max_noise=0.0748, min_noise=0.0):
        super(LoadRawDeno, self).__init__()
        self.patch_size = patch_size
        self.data_lists = []
        self.max_noise = max_noise
        self.min_noise = min_noise
        self.raw_stack = DownsamplingShuffle(2)

        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        rgb = cv2.cvtColor(cv2.imread(self.data_lists[index]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = data_aug(rgb, mode=np.random.randint(0, 8))
        h, w, c = rgb.shape
        h = h // 2 * 2
        w = w // 2 * 2
        rgb = rgb[0:h, 0:w, :]

        rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))).float()
        raw = rgb2raw(rgb, is_tensor=True)

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        raw = raw[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]
        # raw_input + noise
        noise = torch.randn([1, 1, self.patch_size, self.patch_size]).mul_(noise_level)
        raw4 = raw + noise
        raw4 = self.raw_stack(raw4.view(1,1,self.patch_size, self.patch_size))
        # cat noise_map
        noise_map = torch.ones([1, 1, self.patch_size//2, self.patch_size//2])*noise_level
        raw4 = torch.cat((raw4, noise_map), 1)

        data = {}
        data['input'] = torch.clamp(raw4[0], 0., 1.)
        data['gt'] = torch.clamp(raw, 0., 1.)

        return data


class LoadRgbDeno(data.Dataset):
    """
    load training simulated datasets

    """
    def __init__(self, data_list, patch_size=64, max_noise=0.0748, min_noise=0.0):
        super(LoadRgbDeno, self).__init__()
        self.patch_size = patch_size
        self.data_lists = []
        self.max_noise = max_noise
        self.min_noise = min_noise

        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        rgb = cv2.cvtColor(cv2.imread(self.data_lists[index]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = data_aug(rgb, mode=np.random.randint(0, 8))
        h, w, c = rgb.shape
        h = h // 2 * 2
        w = w // 2 * 2
        rgb = rgb[0:h, 0:w, :]
        rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))).float()

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        rgb = rgb[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]
        # raw_input + noise
        noise = torch.randn([1, 1, self.patch_size, self.patch_size]).mul_(noise_level)
        rgb_noisy = rgb + noise

        # cat noise_map
        noise_map = torch.ones([1, 1, self.patch_size, self.patch_size])*noise_level
        rgb_noisy = torch.cat((rgb_noisy, noise_map), 1)[0]

        data = {}
        data['input'] = torch.clamp(rgb_noisy, 0., 1.)
        data['gt'] = torch.clamp(rgb, 0., 1.)

        return data



class LoadRawSR(data.Dataset):
    """
    load training simulated datasets

    """
    def __init__(self, data_list, patch_size=64, scale=2, denoise=False, max_noise=0.0748, min_noise=0.0,
                 downsampler='avg'):
        super(LoadRawSR, self).__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.data_lists = []
        self.denoise = denoise
        self.max_noise = max_noise
        self.min_noise = min_noise
        self.downsampler = downsampler
        self.raw_stack = DownsamplingShuffle(2)

        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        rgb = cv2.cvtColor(cv2.imread(self.data_lists[index]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = data_aug(rgb, mode=np.random.randint(0, 8))
        h, w, c = rgb.shape
        h = h // 8 * 8
        w = w // 8 * 8
        rgb = rgb[0:h, 0:w, :]

        if self.downsampler == 'bic':
            lr_rgb = cv2.resize(rgb.copy(), (0,0), fx=1/self.scale, fy=1/self.scale, interpolation=cv2.INTER_CUBIC)
            lr_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(lr_rgb, [2, 0, 1]))).float()

        # rgb = rgb.astype(np.float32) / 255.
        rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))).float()

        if self.downsampler == 'avg':
            lr_rgb = F.avg_pool2d(rgb.clone(), self.scale, self.scale)

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        rgb = rgb[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        lr_rgb = lr_rgb[:, hi//self.scale: hi//self.scale + self.patch_size//self.scale,
                 wi//self.scale: wi//self.scale + self.patch_size//self.scale]
        lr_raw = rgb2raw(lr_rgb, is_tensor=True)
        raw = rgb2raw(rgb, is_tensor=True)

        lr_raw = lr_raw.view(1, 1, lr_raw.shape[-2], lr_raw.shape[-1])
        lr_raw = self.raw_stack(lr_raw)

        if self.denoise:
            noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]

            # raw_input + noise
            noise = torch.randn([1, 1, self.patch_size//self.scale, self.patch_size//self.scale]).mul_(noise_level)
            lr_raw = lr_raw + self.raw_stack(noise)

            # cat noise_map
            noise_map = torch.ones([1, 1, self.patch_size//(2*self.scale), self.patch_size//(2*self.scale)])*noise_level
            lr_raw = torch.cat((lr_raw, noise_map), 1)


        data = {}
        data['input'] = torch.clamp(lr_raw[0], 0., 1.)
        data['gt'] = torch.clamp(raw, 0., 1.)

        return data


class LoadRgbSR(data.Dataset):
    """
    load training simulated datasets

    """
    def __init__(self, data_list, patch_size=64, scale=2, denoise=False, max_noise=0.0748, min_noise=0.0,
                 downsampler='avg'):
        super(LoadRgbSR, self).__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.data_lists = []
        self.denoise = denoise
        self.max_noise = max_noise
        self.min_noise = min_noise
        self.downsampler = downsampler

        # read image list from txt
        fin = open(data_list)
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            self.data_lists.append(line[0])
        fin.close()

    def __len__(self):
        return len(self.data_lists)

    def __getitem__(self, index):
        # read images, crop size
        rgb = cv2.cvtColor(cv2.imread(self.data_lists[index]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = data_aug(rgb, mode=np.random.randint(0, 8))
        h, w, c = rgb.shape
        h = h // 2 * 2
        w = w // 2 * 2
        rgb = rgb[0:h, 0:w, :]

        if self.downsampler == 'bic':
            lr_rgb = cv2.resize(rgb.copy(), (0,0), fx=1/self.scale, fy=1/self.scale, interpolation=cv2.INTER_CUBIC)
            lr_rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(lr_rgb, [2, 0, 1]))).float()

        rgb = torch.from_numpy(np.ascontiguousarray(np.transpose(rgb, [2, 0, 1]))).float()

        if self.downsampler == 'avg':
            lr_rgb = F.avg_pool2d(rgb.clone(), self.scale, self.scale)

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%2
        hi = hi - hi%2
        # wi, hi, start point in gt

        rgb = rgb[:, hi: hi + self.patch_size, wi: wi + self.patch_size]
        lr_rgb = lr_rgb[:, hi//self.scale: hi//self.scale + self.patch_size//self.scale, wi//self.scale: wi//self.scale + self.patch_size//self.scale]
        if self.denoise:
            noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]
            # raw_input + noise
            noise = torch.randn([1, 1, self.patch_size//self.scale, self.patch_size//self.scale]).mul_(noise_level)
            lr_rgb = lr_rgb + noise

            # cat noise_map
            noise_map = torch.ones([1, 1, self.patch_size//self.scale, self.patch_size//self.scale])*noise_level
            lr_rgb = torch.cat((lr_rgb, noise_map), 1)[0]


        data = {}
        data['input'] = torch.clamp(lr_rgb, 0., 1.)
        data['gt'] = torch.clamp(rgb, 0., 1.)

        return data

# class LoadRawDenoise(data.Dataset):
#     """
#     load datasets for raw denoising
#
#     """
#     def __init__(self, data_list, patch_num=1, patch_height=100, patch_width=100,
#                  max_noise=0.0748, min_noise=0.0):
#         super(LoadRawDenoise, self).__init__()
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.patch_num = patch_num
#         self.rgb_targets = []
#         self.max_noise = max_noise
#         self.min_noise = min_noise
#
#         # read image list from txt
#         fin = open(data_list)
#         lines = fin.readlines()
#         for line in lines:
#             line = line.strip().split()
#             self.rgb_targets.append(line[0])
#         fin.close()
#
#     def __len__(self):
#         return len(self.rgb_targets)
#
#     def __getitem__(self, index):
#         # read images, crop size
#         rgb = _read_image(self.rgb_targets[index])
#         h, w, c = rgb.shape
#         # h = h // 8 * 8
#         # w = w // 8 * 8
#         # rgb = rgb[0:h, 0:w, :]
#
#         raw = _img2raw(rgb.copy(), 'rggb')
#
#         # numpy to torch
#         raw = torch.from_numpy(raw).float()
#         raw = raw.contiguous().view(-1, 1, h, w)
#         down = DownsamplingShuffle(2)
#         raw = down(raw)
#
#         # crop input, gt patches ( overlapping)
#         wi = random.randint(0, w // 2 - self.patch_width)
#         hi = random.randint(0, h // 2 - self.patch_height)
#         # wi, hi, start point in raw_input
#
#         raw_patches = raw[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#         for j in range(self.patch_num - 1):
#             wi = random.randint(0, w // 2 - self.patch_width)
#             hi = random.randint(0, h // 2 - self.patch_height)
#
#             raw_patch = raw[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#             raw_patches = torch.cat((raw_patches, raw_patch), 0)
#
#         raw_input_patches = torch.FloatStorage.clone(raw_patches)
#         # add noise
#         noise_level = max(self.min_noise, np.random.rand(1) * self.max_noise)[0]
#         # raw_input + noise
#         raw_input_patches = add_noise_raw(raw_input_patches, noise_level)
#         # cat noise_map
#         noise_map = torch.ones(
#             [raw_input_patches.shape[0], 1, raw_input_patches.shape[-2], raw_input_patches.shape[-1]]) * noise_level
#         raw_input_patches = torch.cat((raw_input_patches, noise_map), 1)
#
#         del raw, rgb
#         return raw_input_patches.float().clamp(0.0, 1.0), raw_patches.float().clamp(0.0, 1.0)
#
# class LoadDemo(data.Dataset):
#     """
#     load training analogous datasets
#
#     """
#     def __init__(self, data_list, patch_num=1, patch_height=100, patch_width=100,
#                  denoise=False, max_noise=0.0748, min_noise=0.0):
#         super(LoadDemo, self).__init__()
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.patch_num = patch_num
#         self.rgb_targets = []
#         self.denoise = denoise
#         self.max_noise = max_noise
#         self.min_noise = min_noise
#
#         # read image list from txt
#         fin = open(data_list)
#         lines = fin.readlines()
#         for line in lines:
#             line = line.strip().split()
#             self.rgb_targets.append(line[0])
#         fin.close()
#
#     def __len__(self):
#         return len(self.rgb_targets)
#
#     def __getitem__(self, index):
#         # read images, crop size
#         rgb = _read_image(self.rgb_targets[index])
#         h, w, c = rgb.shape
#         h = h // 8 * 8
#         w = w // 8 * 8
#         rgb = rgb[0:h, 0:w, :]
#
#         raw = _img2raw(rgb.copy(), 'rggb')
#
#         # numpy to torch
#         rgb = torch.from_numpy(np.transpose(rgb, [2, 0, 1])).float()
#         rgb = rgb.contiguous().view(-1, 3, h, w)
#         raw = torch.from_numpy(raw).float()
#         raw = raw.contiguous().view(-1, 1, h, w)
#         down = DownsamplingShuffle(2)
#         raw_input = down(raw)
#
#         # crop input, gt patches ( overlapping)
#         wi = random.randint(0, w // 2 - self.patch_width)
#         hi = random.randint(0, h // 2 - self.patch_height)
#         # wi, hi, start point in raw_input
#
#         rgb_patches = rgb[:, :, (hi * 2): (hi + self.patch_height) * 2,
#                           wi * 2: (wi + self.patch_width) * 2]
#         raw_input_patches = raw_input[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#
#         for j in range(self.patch_num - 1):
#             wi = random.randint(0, w // 2 - self.patch_width)
#             hi = random.randint(0, h // 2 - self.patch_height)
#             rgb_patch = rgb[:, :, (hi * 2): (hi + self.patch_height) * 2,
#                                    wi * 2: (wi + self.patch_width) * 2]
#             rgb_patches = torch.cat((rgb_patches, rgb_patch), 0)
#
#             raw_input_patch = raw_input[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#             raw_input_patches = torch.cat((raw_input_patches, raw_input_patch), 0)
#         del raw, rgb
#
#         # add gaussian noise on raw_input
#         if self.denoise:
#             noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]
#
#             # raw_input + noise
#             raw_input_patches = add_noise_raw(raw_input_patches, noise_level)
#             # cat noise_map
#             noise_map = torch.ones([raw_input_patches.shape[0], 1, raw_input_patches.shape[-2], raw_input_patches.shape[-1]])*noise_level
#             raw_input_patches = torch.cat((raw_input_patches, noise_map), 1)
#             del noise_map
#         return raw_input_patches.float().clamp(0.0, 1.0), rgb_patches.float().clamp(0.0, 1.0)
#
#
# class LoadRawDenoise(data.Dataset):
#     """
#     load datasets for raw denoising
#
#     """
#     def __init__(self, data_list, patch_num=1, patch_height=100, patch_width=100,
#                  max_noise=0.0748, min_noise=0.0):
#         super(LoadRawDenoise, self).__init__()
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.patch_num = patch_num
#         self.rgb_targets = []
#         self.max_noise = max_noise
#         self.min_noise = min_noise
#
#         # read image list from txt
#         fin = open(data_list)
#         lines = fin.readlines()
#         for line in lines:
#             line = line.strip().split()
#             self.rgb_targets.append(line[0])
#         fin.close()
#
#     def __len__(self):
#         return len(self.rgb_targets)
#
#     def __getitem__(self, index):
#         # read images, crop size
#         rgb = _read_image(self.rgb_targets[index])
#         h, w, c = rgb.shape
#         # h = h // 8 * 8
#         # w = w // 8 * 8
#         # rgb = rgb[0:h, 0:w, :]
#
#         raw = _img2raw(rgb.copy(), 'rggb')
#
#         # numpy to torch
#         raw = torch.from_numpy(raw).float()
#         raw = raw.contiguous().view(-1, 1, h, w)
#         down = DownsamplingShuffle(2)
#         raw = down(raw)
#
#         # crop input, gt patches ( overlapping)
#         wi = random.randint(0, w // 2 - self.patch_width)
#         hi = random.randint(0, h // 2 - self.patch_height)
#         # wi, hi, start point in raw_input
#
#         raw_patches = raw[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#         for j in range(self.patch_num - 1):
#             wi = random.randint(0, w // 2 - self.patch_width)
#             hi = random.randint(0, h // 2 - self.patch_height)
#
#             raw_patch = raw[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#             raw_patches = torch.cat((raw_patches, raw_patch), 0)
#
#         raw_input_patches = torch.FloatStorage.clone(raw_patches)
#         # add noise
#         noise_level = max(self.min_noise, np.random.rand(1) * self.max_noise)[0]
#         # raw_input + noise
#         raw_input_patches = add_noise_raw(raw_input_patches, noise_level)
#         # cat noise_map
#         noise_map = torch.ones(
#             [raw_input_patches.shape[0], 1, raw_input_patches.shape[-2], raw_input_patches.shape[-1]]) * noise_level
#         raw_input_patches = torch.cat((raw_input_patches, noise_map), 1)
#
#         del raw, rgb
#         return raw_input_patches.float().clamp(0.0, 1.0), raw_patches.float().clamp(0.0, 1.0)
#
#
# class LoadRGBDenoise(data.Dataset):
#     """
#     load datasets for denoising on RGB
#
#     """
#     def __init__(self, data_list, patch_num=1, patch_height=100, patch_width=100,
#                  max_noise=0.0748, min_noise=0.0):
#         super(LoadRGBDenoise, self).__init__()
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.patch_num = patch_num
#         self.rgb_targets = []
#         self.max_noise = max_noise
#         self.min_noise = min_noise
#
#         # read image list from txt
#         fin = open(data_list)
#         lines = fin.readlines()
#         for line in lines:
#             line = line.strip().split()
#             self.rgb_targets.append(line[0])
#         fin.close()
#
#     def __len__(self):
#         return len(self.rgb_targets)
#
#     def __getitem__(self, index):
#         # read images, crop size
#         rgb = _read_image(self.rgb_targets[index])
#         h, w, c = rgb.shape
#         h = h // 8 * 8
#         w = w // 8 * 8
#         rgb = rgb[0:h, 0:w, :]
#
#         # numpy to torch
#         rgb = torch.from_numpy(np.transpose(rgb, [2, 0, 1])).float()
#         rgb = rgb.contiguous().view(-1, 3, h, w)
#
#         # crop input, gt patches ( overlapping)
#         wi = random.randint(0, w - self.patch_width)
#         hi = random.randint(0, h - self.patch_height)
#         # wi, hi, start point in raw_input
#
#         rgb_patches = rgb[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#         for j in range(self.patch_num - 1):
#             wi = random.randint(0, w - self.patch_width)
#             hi = random.randint(0, h - self.patch_height)
#
#             rgb_patch = rgb[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#             rgb_patches = torch.cat((rgb_patches, rgb_patch), 0)
#
#         rgb_input_patches = torch.FloatStorage.clone(rgb_patches)
#         # add gaussian noise on rgb_input
#         noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]
#         # raw_input + noise
#         rgb_input_patches = add_noise(rgb_input_patches, noise_level)
#         # cat noise_map
#         noise_map = torch.ones([rgb_input_patches.shape[0], 1, rgb_input_patches.shape[-2], rgb_input_patches.shape[-1]])*noise_level
#         rgb_input_patches = torch.cat((rgb_input_patches, noise_map), 1)
#
#         return rgb_input_patches.float().clamp(0.0, 1.0), rgb_patches.float().clamp(0.0, 1.0)
#
#
# class LoadSRRGB(data.Dataset):
#     """
#     load datasets for SR
#
#     """
#     def __init__(self, data_list, patch_num=1, patch_height=100, patch_width=100,
#                  denoise=False, max_noise=0.0748, min_noise=0.0, scale=2, downsampler='avg'):
#         super(LoadSRRGB, self).__init__()
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.patch_num = patch_num
#         self.rgb_targets = []
#         self.max_noise = max_noise
#         self.min_noise = min_noise
#         self.scale = scale
#         self.downsampler = downsampler
#         self.denoise = denoise
#
#         # read image list from txt
#         fin = open(data_list)
#         lines = fin.readlines()
#         for line in lines:
#             line = line.strip().split()
#             self.rgb_targets.append(line[0])
#         fin.close()
#
#     def __len__(self):
#         return len(self.rgb_targets)
#
#     def __getitem__(self, index):
#         # read images, crop size
#         rgb = _read_image(self.rgb_targets[index])
#         h, w, c = rgb.shape
#         h = h // 2 * 2
#         w = w // 2 * 2
#         rgb = rgb[0:h, 0:w, :]
#         if self.downsampler == 'bicubic':
#             rgb_down = imresize(rgb, 1/float(self.scale), interp=self.downsampler)
#             rgb_down = torch.from_numpy(np.transpose(rgb_down, [2, 0, 1])).float()
#             rgb_down = rgb_down.contiguous().view(-1, 3, h//self.scale, w//self.scale)
#         # numpy to torch
#
#         rgb = torch.from_numpy(np.transpose(rgb, [2, 0, 1])).float()
#         rgb = rgb.contiguous().view(-1, 3, h, w)
#
#         if self.downsampler =='avg':
#             rgb_input = rgb_avg_downsample(rgb, self.scale)
#
#         # crop input, gt patches ( overlapping)
#         wi = random.randint(0, w//self.scale - self.patch_width)
#         hi = random.randint(0, h//self.scale - self.patch_height)
#         # wi, hi, start point in raw_input
#
#         rgb_input_patches = rgb_input[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#         rgb_target_patches = rgb[:, :, (hi * self.scale): (hi + self.patch_height) * self.scale,
#                                  wi * self.scale: (wi + self.patch_width) * self.scale]
#         for j in range(self.patch_num - 1):
#             wi = random.randint(0, w//self.scale - self.patch_width)
#             hi = random.randint(0, h//self.scale - self.patch_height)
#             rgb_target_patch = rgb[:, :, (hi * self.scale): (hi + self.patch_height) * self.scale,
#                                    wi * self.scale: (wi + self.patch_width) * self.scale]
#             rgb_target_patches = torch.cat((rgb_target_patches, rgb_target_patch), 0)
#
#             rgb_input_patch = rgb_input[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#             rgb_input_patches = torch.cat((rgb_input_patches, rgb_input_patch), 0)
#
#         if self.denoise:
#             # add gaussian noise on rgb_input
#             noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]
#             # raw_input + noise
#             rgb_input_patches = add_noise(rgb_input_patches, noise_level)
#             # cat noise_map
#             noise_map = torch.ones([rgb_input_patches.shape[0], 1, rgb_input_patches.shape[-2], rgb_input_patches.shape[-1]])*noise_level
#             rgb_input_patches = torch.cat((rgb_input_patches, noise_map), 1)
#
#         return rgb_input_patches.float().clamp(0.0, 1.0), rgb_target_patches.float().clamp(0.0, 1.0)
#
#
# class LoadSRRAW(data.Dataset):
#     """
#     load datasets for SR
#
#     """
#     def __init__(self, data_list, patch_num=1, patch_height=100, patch_width=100,
#                  denoise=False, max_noise=0.0748, min_noise=0.0, scale=2, downsampler='avg'):
#         super(LoadSRRAW, self).__init__()
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.patch_num = patch_num
#         self.rgb_targets = []
#         self.max_noise = max_noise
#         self.min_noise = min_noise
#         self.scale = scale
#         self.downsampler = downsampler
#         self.denoise = denoise
#
#         # read image list from txt
#         fin = open(data_list)
#         lines = fin.readlines()
#         for line in lines:
#             line = line.strip().split()
#             self.rgb_targets.append(line[0])
#         fin.close()
#
#     def __len__(self):
#         return len(self.rgb_targets)
#
#     def __getitem__(self, index):
#         # read images, crop size
#         rgb = _read_image(self.rgb_targets[index])
#         h, w, c = rgb.shape
#         h = h // 8 * 8
#         w = w // 8 * 8
#         rgb = rgb[0:h, 0:w, :]
#         raw = _img2raw(rgb.copy(), 'rggb')
#         # numpy to torch
#         rgb = torch.from_numpy(np.transpose(rgb, [2, 0, 1])).float()
#         rgb = rgb.contiguous().view(-1, 3, h, w)
#         raw = torch.from_numpy(raw).float()
#         raw = raw.contiguous().view(-1, 1, h, w)
#
#         # raw input. down-sampling: ps binning
#         if self.downsampler == 'avg':
#             raw_input = _pixelshift_binning(torch.FloatStorage.clone(rgb), scale=self.scale)
#         # else
#         # raw_input = bicubic
#
#         # crop input, gt patches ( overlapping)
#         wi = random.randint(0, w // (self.scale * 2) - self.patch_width)
#         hi = random.randint(0, h // (self.scale * 2) - self.patch_height)
#         # wi, hi, start point in raw_input
#
#         raw_target_patches = raw[:, :, hi * self.scale * 2: (hi + self.patch_height) * self.scale * 2,
#                              wi * self.scale * 2: (wi + self.patch_width) * self.scale * 2]
#         raw_input_patches = raw_input[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#
#         for j in range(self.patch_num - 1):
#             wi = random.randint(0, w // (self.scale * 2) - self.patch_width)
#             hi = random.randint(0, h // (self.scale * 2) - self.patch_height)
#             raw_target_patch = raw[:, :, hi * self.scale * 2: (hi + self.patch_height) * self.scale * 2,
#                                wi * self.scale * 2: (wi + self.patch_width) * self.scale * 2]
#             raw_target_patches = torch.cat((raw_target_patches, raw_target_patch), 0)
#             raw_input_patch = raw_input[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#             raw_input_patches = torch.cat((raw_input_patches, raw_input_patch), 0)
#
#         down = DownsamplingShuffle(2)
#         raw_target_patches = down(raw_target_patches)
#
#         # add gaussian noise on raw_input
#         if self.denoise:
#             noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]
#             # raw_input + noise
#             raw_input_patches = add_noise_raw(raw_input_patches, noise_level)
#
#             # cat noise_map
#             noise_map = torch.ones([raw_input_patches.shape[0], 1, raw_input_patches.shape[-2], raw_input_patches.shape[-1]])*noise_level
#             raw_input_patches = torch.cat((raw_input_patches, noise_map), 1)
#
#         return raw_input_patches.float().clamp(0.0, 1.0), raw_target_patches.float().clamp(0.0, 1.0)
#

# class LoadJointMat2Label(data.Dataset):
#     """
#     load training analogous datasets
#
#     """
#     def __init__(self, data_list, patch_num=1, patch_height=100, patch_width=100, scale=2,
#                  denoise=False, max_noise=0.0748, min_noise=0.0, downsampler='avg'):
#         super(LoadJointMat2Label, self).__init__()
#         self.scale = scale
#         self.patch_height = patch_height
#         self.patch_width = patch_width
#         self.patch_num = patch_num
#         self.rgb_targets = []
#         self.denoise = denoise
#         self.max_noise = max_noise
#         self.min_noise = min_noise
#         self.downsampler = downsampler
#         self.avgpool = torch.nn.AvgPool2d(scale, stride=scale)
#         self.raw_down_sample = DownsamplingShuffle(2)
#         # read image list from txt
#         fin = open(data_list)
#         lines = fin.readlines()
#         for line in lines:
#             line = line.strip().split()
#             self.rgb_targets.append(line[0])
#         fin.close()
#
#     def __len__(self):
#         return len(self.rgb_targets)
#
#     def __getitem__(self, index):
#         # read images, crop size
#         rggb = np.asarray(loadmat(self.rgb_targets[index])['ps']).astype(np.float32)/32767.
#         rggb = data_aug(rggb, mode=np.random.randint(0, 8))
#         # rgb = _read_image(self.rgb_targets[index])
#         h, w, c = rggb.shape
#
#         raw_input, raw, rgb = rggb_prepro(rggb, self.scale, self.avgpool , self.raw_down_sample)
#
#         # crop input, gt patches ( overlapping)
#         wi = random.randint(0, w // (self.scale * 2) - self.patch_width)
#         hi = random.randint(0, h // (self.scale * 2) - self.patch_height)
#         # wi, hi, start point in raw_input
#
#         raw_target_patches = raw[:, :, hi * self.scale * 2: (hi + self.patch_height) * self.scale * 2,
#                              wi * self.scale * 2: (wi + self.patch_width) * self.scale * 2]
#         rgb_target_patches = rgb[:, :, (hi * self.scale * 2): (hi + self.patch_height) * self.scale * 2,
#                              wi * self.scale * 2: (wi + self.patch_width) * self.scale * 2]
#         raw_input_patches = raw_input[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#
#         for j in range(self.patch_num - 1):
#             wi = random.randint(0, w // (self.scale * 2) - self.patch_width)
#             hi = random.randint(0, h // (self.scale * 2) - self.patch_height)
#             raw_target_patch = raw[:, :, hi * self.scale * 2: (hi + self.patch_height) * self.scale * 2,
#                                wi * self.scale * 2: (wi + self.patch_width) * self.scale * 2]
#             raw_target_patches = torch.cat((raw_target_patches, raw_target_patch), 0)
#             rgb_target_patch = rgb[:, :, (hi * self.scale * 2): (hi + self.patch_height) * self.scale * 2,
#                                wi * self.scale * 2: (wi + self.patch_width) * self.scale * 2]
#             rgb_target_patches = torch.cat((rgb_target_patches, rgb_target_patch), 0)
#             raw_input_patch = raw_input[:, :, hi: hi + self.patch_height, wi: wi + self.patch_width]
#             raw_input_patches = torch.cat((raw_input_patches, raw_input_patch), 0)
#         # del raw, rgb, raw_input
#
#         if self.denoise:
#             noise_level = max(self.min_noise, np.random.rand(1)*self.max_noise)[0]
#
#             # raw_input + noise
#             raw_input_patches = add_noise_raw(raw_input_patches, noise_level)
#
#             # cat noise_map
#             noise_map = torch.ones([raw_input_patches.shape[0], 1, raw_input_patches.shape[-2], raw_input_patches.shape[-1]])*noise_level
#             raw_input_patches = torch.cat((raw_input_patches, noise_map), 1)
#
#         return raw_input_patches.float().clamp(0.0, 1.0), raw_target_patches.float().clamp(0.0, 1.0), \
#                rgb_target_patches.float().clamp(0.0, 1.0)
#
if __name__ == '__main__':
    data_list = '/home/likewise-open/SENSETIME/qianguocheng/Documents' \
                '/CodesProjects/pytorch/p1_small_v1/datasets/train.txt'
    datasets = LoadSRRAW(data_list)
    imgs, raw_gt, rgb_gt = datasets.__getitem__(5)
    print(imgs.shape)


