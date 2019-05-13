import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.common import DownsamplingShuffle
import pdb


def _sigmoid_to_tanh(x):
    """
    range [0, 1] to range [-1, 1]
    :param x: tensor type
    :return: tensor
    """
    return (x - 0.5) * 2.


def _tanh_to_sigmoid(x):
    """
    range [-1, 1] to range [0, 1]
    :param x:
    :return:
    """
    return x * 0.5 + 0.5


def _255_to_tanh(x):
    """
    range [0, 255] to range [-1, 1]
    :param x:
    :return:
    """
    return (x - 127.5) / 127.5


def _tanh_to_255(x):
    """
    range [-1. 1] to range [0, 255]
    :param x:
    :return:
    """
    return x * 127.5 + 127.5


# TODO: _sigmoid_to_255(x), _255_to_sigmoid(x)
# def _sigmoid_to_255(x):
# def _255_to_sigmoid(x):


'''
raw image processing 

'''
def rgb2raw(img, is_tensor = False, bayer_pattern='rggb'):
    """
    :param img:
    :param bayer_pattern:
    :return: generate raw image from rgb
    """
    if bayer_pattern == 'rggb':
        h_shift = 0
        w_shift = 0
    elif bayer_pattern == 'grbg':
        h_shift = 0
        w_shift = 1
    elif bayer_pattern == 'gbrg':
        h_shift = 1
        w_shift = 0
    elif bayer_pattern == 'bggr':
        h_shift = 1
        w_shift = 1
    else:
        raise SystemExit('bayer_pattern is not supported')


    if not is_tensor:
        raw = img[:, :, 1:2]
        h, w, c = img.shape
        raw[h_shift:h:2, w_shift:w:2, :] = img[h_shift:h:2, w_shift:w:2, 0:1]
        raw[1 - h_shift:h:2, 1 - w_shift:w:2, :] = img[1 - h_shift:h:2, 1 - w_shift:w:2, 2:3]
    else:
        raw = img[1:2,:,:]
        c, h, w = img.shape
        raw[:, h_shift:h:2, w_shift:w:2] = img[0:1, h_shift:h:2, w_shift:w:2]
        raw[:, 1-h_shift:h:2, 1-w_shift:w:2] = img[2:3, 1-h_shift:h:2, 1-w_shift:w:2]

    return raw

'''
raw image processing 

'''
def rggb_prepro(img, scale):
    """
    :param img:
    :param scale:
    :return: raw_input, raw, rgb
    """

    # raw
    h, w, c = img.shape
    # rgb
    g = 0.5*(img[:,:,1] + img[:,:,2])
    rgb = torch.from_numpy(np.stack((img[:,:,0], g, img[:,:,3]), axis=0))

    rggb = torch.from_numpy(np.ascontiguousarray(np.transpose(img, [2, 0, 1])))
    lr_rggb = F.avg_pool2d(rggb.clone(), scale, scale)

    lr_raw = lr_rggb[1:2, :, :]
    lr_raw[0, 0:h:2, 0:w:2] = lr_rggb[0, 0:h:2, 0:w:2]  # r
    lr_raw[0, 1:h:2, 0:w:2] = lr_rggb[2, 1:h:2, 0:w:2]  # g2
    lr_raw[0, 1:h:2, 1:w:2] = lr_rggb[3, 1:h:2, 1:w:2]  # b

    raw = rggb[1:2,:,:]
    raw[0, 0:h:2, 0:w:2] = rggb[0, 0:h:2, 0:w:2]
    raw[0, 1:h:2, 0:w:2] = rggb[2, 1:h:2, 0:w:2]
    raw[0, 1:h:2, 1:w:2] = rggb[3, 1:h:2, 1:w:2]

    return lr_raw, raw, rgb


def rgb_avg_downsample(rgb, scale=2):
    """
    :param rgb:rgb
    :param scale:scale
    :return: down-sampled rgb
    """
    if len(rgb.shape) < 4:
        rgb = torch.from_numpy(np.transpose(rgb, [2, 0, 1]))
        c, h, w = rgb.shape
        rgb = rgb.contiguous().view(-1, 3, h, w)

    b, c, h, w = rgb.shape

    # AvgPool2d

    avgpool = nn.AvgPool2d(scale, stride=scale)

    return avgpool(rgb)


def generate_gaussian_noise(h, w, noise_level):

    return torch.randn([1, h, w]).mul_(noise_level)


# def add_noise(img, noise_level):
#
#     noise = torch.randn(img.size()).mul_(noise_level)
#     return img + noise


def rgb_downsample(img, downsampler, scale):
    """
    :param img: (batch, 3, h, w)
    :param downsampler: bicubic, avg
    :return: noisy raw
    """
    noise = np.random.normal(scale=noise_level, size=img.shape)
    noise = torch.from_numpy(noise).float()
    return img + noise


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

