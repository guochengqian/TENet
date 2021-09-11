import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
from bicubic_pytorch import core


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


def rggb_prepro(img, scale):
    """
    :param img:
    :param scale:
    :return: raw_input, raw, rgb
    """

    # raw
    h, w, c = img.shape
    # rgb
    g = 0.5 * (img[:, :, 1] + img[:, :, 2])
    rgb = torch.from_numpy(np.stack((img[:, :, 0], g, img[:, :, 3]), axis=0))

    rggb = torch.from_numpy(np.ascontiguousarray(np.transpose(img, [2, 0, 1])))
    lr_rggb = F.avg_pool2d(rggb.clone(), scale, scale)

    lr_raw = lr_rggb[1:2, :, :]
    lr_raw[0, 0:h:2, 0:w:2] = lr_rggb[0, 0:h:2, 0:w:2]  # r
    lr_raw[0, 1:h:2, 0:w:2] = lr_rggb[2, 1:h:2, 0:w:2]  # g2
    lr_raw[0, 1:h:2, 1:w:2] = lr_rggb[3, 1:h:2, 1:w:2]  # b

    raw = rggb[1:2, :, :]
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


def aug_img_np(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.ascontiguousarray(np.flipud(img))
    elif mode == 2:
        return np.ascontiguousarray(np.rot90(img))
    elif mode == 3:
        return np.ascontiguousarray(np.flipud(np.rot90(img)))
    elif mode == 4:
        return np.ascontiguousarray(np.rot90(img, k=2))
    elif mode == 5:
        return np.ascontiguousarray(np.flipud(np.rot90(img, k=2)))
    elif mode == 6:
        return np.ascontiguousarray(np.rot90(img, k=3))
    elif mode == 7:
        return np.ascontiguousarray(np.flipud(np.rot90(img, k=3)))


def aug_img(img):
    """
    :param img: input PIL image.
    :return PIL image augmentation
    """
    hflip = random.random() < 0.5
    if hflip:
        img = TF.hflip(img)
    vflip = random.random() < 0.5
    if vflip:
        img = TF.vflip(img)
    rotate = random.random() < 0.5
    if rotate:
        degree = random.choice([-90, 90, 180])
        img = TF.rotate(img, degree)
    return img


def crop_img_np(rgb, patch_size, center_crop=True):
    """
    :param rgb: input numpy image. [H, W, C]
    :param patch_size: desired patch_size
    :param center_crop:
    :return: crop img in PIL
    """
    # crop
    w, h, _ = rgb.shape
    if not (w, h) == (patch_size, patch_size):
        if not center_crop:
            i = random.randint(0, h - patch_size)
            j = random.randint(0, w - patch_size)
        else:
            i = h//2 - patch_size//2
            j = w//2 - patch_size//2
        rgb = rgb[i:i+patch_size, j:j+patch_size, :]
    return rgb


def crop_img(rgb, patch_size, center_crop=True):
    """
    :param rgb: input PIL image.
    :param patch_size: desired patch_size
    :param center_crop:
    :return: crop img in PIL
    """
    # crop
    w, h = rgb.size
    if not (w, h) == (patch_size, patch_size):
        if not center_crop:
            i = random.randint(0, h - patch_size)
            j = random.randint(0, w - patch_size)
            rgb = TF.crop(rgb, i, j, patch_size, patch_size)
        else:
            rgb = TF.center_crop(rgb, patch_size)

    return rgb


def downsample_tensor(img, scale=2., downsampler='bic'):
    """
    :param img: the tensor format of a mini-batch of img
    :return: tensor list (lr_img, rgb)
    """
    if downsampler == 'bic':
        lr_img = core.imresize(img, scale=1 / scale)
    elif downsampler == 'avg':
        lr_img = F.avg_pool2d(img, scale, scale)
    else:
        return NotImplementedError('{} is not supported'.format(downsampler))
    return lr_img

