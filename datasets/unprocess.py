# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unprocesses sRGB images into realistic raw data.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
import torch.distributions as tdist


def random_ccm():
    """Generates random RGB -> Camera color correction matrices."""
    # Takes a random convex combination of XYZ -> Camera CCMs.
    xyz2cams = [[[1.0234, -0.2969, -0.2266],
                 [-0.5625, 1.6328, -0.0469],
                 [-0.0703, 0.2188, 0.6406]],
                [[0.4913, -0.0541, -0.0202],
                 [-0.613, 1.3513, 0.2906],
                 [-0.1564, 0.2151, 0.7183]],
                [[0.838, -0.263, -0.0639],
                 [-0.2887, 1.0725, 0.2496],
                 [-0.0627, 0.1427, 0.5438]],
                [[0.6596, -0.2079, -0.0562],
                 [-0.4782, 1.3016, 0.1933],
                 [-0.097, 0.1581, 0.5181]]]
    num_ccms = len(xyz2cams)
    xyz2cams = torch.FloatTensor(xyz2cams)
    weights = torch.FloatTensor(num_ccms, 1, 1).uniform_(1e-8, 1e8)
    weights_sum = torch.sum(weights, dim=0)
    xyz2cam = torch.sum(xyz2cams * weights, dim=0) / weights_sum

    # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)

    # Normalizes each row.
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    return rgb2cam


def random_gains():
    """Generates random gains for brightening and white balance."""
    # RGB gain represents brightening.
    n = tdist.Normal(loc=torch.tensor([0.8]), scale=torch.tensor([0.1]))
    rgb_gain = 1.0 / n.sample()

    # Red and blue gains represent white balance.
    red_gain = torch.FloatTensor(1).uniform_(1.9, 2.4)
    blue_gain = torch.FloatTensor(1).uniform_(1.5, 1.9)
    return rgb_gain, red_gain, blue_gain


def inverse_smoothstep(image):
    """Approximately inverts a global tone mapping curve."""
    image = torch.clamp(image, min=0.0, max=1.0)
    out = 0.5 - torch.sin(torch.asin(1.0 - 2.0 * image) / 3.0)
    return out


def gamma_expansion(image):
    """Converts from gamma to linear space."""
    # Clamps to prevent numerical instability of gradients near zero.
    out = torch.clamp(image, min=1e-8) ** 2.2
    return out


def apply_ccm(image, ccm):
    """Applies a color correction matrix."""
    shape = image.size()
    img = image.clone().reshape(3, -1)
    out = torch.matmul(ccm, img).reshape(shape)
    return out


# todo: can make it compact
def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
    """Inverts gains while safely handling saturated pixels."""
    image = image.permute(1, 2, 0)  # Permute the image tensor to HxWxC format from CxHxW format
    gains = torch.stack((1.0 / red_gain, torch.tensor([1.0]), 1.0 / blue_gain)) / rgb_gain
    gains = gains.squeeze()
    gains = gains[None, None, :]
    # Prevents dimming of saturated pixels by smoothly masking gains near white.
    gray = torch.mean(image, dim=-1, keepdim=True)
    inflection = 0.9
    mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0
    safe_gains = torch.max(mask + (1.0 - mask) * gains, gains)
    out = image * safe_gains
    out = out.permute(2, 0, 1)  # Re-Permute the tensor back to CxHxW format
    return out


def mosaic(image):
    """Extracts RGGB Bayer planes from an RGB image."""
    red = image[0, 0::2, 0::2]
    green_red = image[1, 0::2, 1::2]
    green_blue = image[1, 1::2, 0::2]
    blue = image[2, 1::2, 1::2]
    out = torch.stack((red, green_red, green_blue, blue), dim=0)
    return out

# def mosaic(img, is_tensor=True, bayer_pattern='rggb'):
#     """
#     :param img:
#     :param is_tensor: the image is in tensor [HxWxC]
#     :param bayer_pattern: RGGB
#     :return: generate bayer raw image from rgb
#     """
#     if bayer_pattern == 'rggb':
#         h_shift = 0
#         w_shift = 0
#     elif bayer_pattern == 'grbg':
#         h_shift = 0
#         w_shift = 1
#     elif bayer_pattern == 'gbrg':
#         h_shift = 1
#         w_shift = 0
#     elif bayer_pattern == 'bggr':
#         h_shift = 1
#         w_shift = 1
#     else:
#         raise SystemExit('bayer_pattern is not supported')
#
#     if not is_tensor:
#         raw = img[:, :, 1:2]
#         h, w, c = img.shape
#         raw[h_shift:h:2, w_shift:w:2, :] = img[h_shift:h:2, w_shift:w:2, 0:1]
#         raw[1 - h_shift:h:2, 1 - w_shift:w:2, :] = img[1 - h_shift:h:2, 1 - w_shift:w:2, 2:3]
#     else:
#         raw = img[1:2, :, :]
#         c, h, w = img.shape
#         raw[:, h_shift:h:2, w_shift:w:2] = img[0:1, h_shift:h:2, w_shift:w:2]
#         raw[:, 1 - h_shift:h:2, 1 - w_shift:w:2] = img[2:3, 1 - h_shift:h:2, 1 - w_shift:w:2]
#
#     return raw


def unprocess(image, rgb2cam, rgb_gain, red_gain, blue_gain):
    """Unprocesses an image from sRGB to realistic raw data."""
    # Approximately inverts global tone mapping.
    image = inverse_smoothstep(image)
    # Inverts gamma compression.
    image = gamma_expansion(image)
    # Inverts color correction.
    image = apply_ccm(image, rgb2cam)
    # Approximately inverts white balance and brightening.
    image = safe_invert_gains(image, rgb_gain, red_gain, blue_gain)
    # Clips saturated pixels.
    lin_rgb = torch.clamp(image, min=0.0, max=1.0)
    # Applies a Bayer mosaic.
    raw = mosaic(lin_rgb.clone())

    return raw, lin_rgb


def random_unprocess(images):
    """Unprocesses an image from sRGB to realistic raw data."""
    # Randomly creates image metadata.
    rgb2cam = random_ccm()
    cam2rgb = torch.inverse(rgb2cam)
    rgb_gain, red_gain, blue_gain = random_gains()
    metadata = {
        'cam2rgb': cam2rgb,
        'rgb_gain': rgb_gain,
        'red_gain': red_gain,
        'blue_gain': blue_gain,
    }
    return [unprocess(img, rgb2cam, rgb_gain, red_gain, blue_gain) for img in images], metadata


def random_noise_levels():
    """Generates random noise levels from a log-log linear distribution."""
    log_min_shot_noise = np.log(0.0001)
    log_max_shot_noise = np.log(0.012)
    log_shot_noise = torch.FloatTensor(1).uniform_(log_min_shot_noise, log_max_shot_noise)
    shot_noise = torch.exp(log_shot_noise)

    line = lambda x: 2.18 * x + 1.20
    n = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26]))
    log_read_noise = line(log_shot_noise) + n.sample()
    read_noise = torch.exp(log_read_noise)
    return shot_noise, read_noise


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
    """Adds random shot (proportional to image) and read (independent) noise."""
    variance = image * shot_noise + read_noise
    n = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
    noise = n.sample()
    out = image + noise
    return out

