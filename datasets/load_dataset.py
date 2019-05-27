import numpy as np
import torch
import torch.utils.data as data
import random
from scipy.io import loadmat
from TorchTools.DataTools.Prepro import rgb2raw, data_aug, rggb_prepro
from model.common import DownsamplingShuffle
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
        rggb = np.asarray(loadmat(self.data_lists[index])['ps']).astype(np.float32)/65535.
        rggb = data_aug(rggb, mode=np.random.randint(0, 8))
        h, w, c = rggb.shape

        lr_raw, raw, rgb = rggb_prepro(rggb.copy(), self.scale)

        # crop gt, keep even
        wi = random.randint(0, w - self.patch_size)
        hi = random.randint(0, h - self.patch_size)
        wi = wi - wi%(self.scale*2)
        hi = hi - hi%(self.scale*2)
        # wi, hi, start point in gt
        # in order to make rggb pattern, hi, hi must be

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
        wi = wi - wi%(self.scale*2)
        hi = hi - hi%(self.scale*2)
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
#



