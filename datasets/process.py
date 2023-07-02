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

"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist


def apply_gains_bayer(bayer_images, red_gains, blue_gains):
    """Applies white balance gains to a batch of Bayer images."""
    red_gains = red_gains.squeeze()
    blue_gains = blue_gains.squeeze()
    red_gains = red_gains.unsqueeze(0) if len(red_gains.shape) == 0 else red_gains
    blue_gains = blue_gains.unsqueeze(0) if len(blue_gains.shape) == 0 else blue_gains
    bayer_images = bayer_images.permute(0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    green_gains = torch.ones_like(red_gains)
    gains = torch.stack([red_gains, green_gains, green_gains, blue_gains], dim=-1)
    gains = gains[:, None, None, :]
    outs = bayer_images * gains
    outs = outs.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return outs


def apply_gains_rgb(rgb, red_gains, blue_gains):
    """Applies white balance gains to a batch of RGB images."""
    red_gains = red_gains.squeeze()
    blue_gains = blue_gains.squeeze()
    red_gains = red_gains.unsqueeze(0) if len(red_gains.shape) == 0 else red_gains
    blue_gains = blue_gains.unsqueeze(0) if len(blue_gains.shape) == 0 else blue_gains
    rgb = rgb.permute(0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    green_gains = torch.ones_like(red_gains)
    gains = torch.stack([red_gains, green_gains, blue_gains], dim=-1)
    gains = gains[:, None, None, :]
    outs = rgb * gains
    outs = outs.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return outs


def demosaic(bayer_images):
    def SpaceToDepth_fact2(x):
        # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        bs = 2
        N, C, H, W = x.size()
        x = x.view(N, C, H // bs, bs, W // bs, bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (bs ** 2), H // bs, W // bs)  # (N, C*bs^2, H//bs, W//bs)
        return x

    def DepthToSpace_fact2(x):
        # From here - https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
        bs = 2
        N, C, H, W = x.size()
        x = x.view(N, bs, bs, C // (bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (bs ** 2), H * bs, W * bs)  # (N, C//bs^2, H * bs, W * bs)
        return x

    """Bilinearly demosaics a batch of RGGB Bayer images."""
    bayer_images = bayer_images.permute(0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format

    shape = bayer_images.size()
    shape = [shape[1] * 2, shape[2] * 2]

    red = bayer_images[Ellipsis, 0:1]
    upsamplebyX = nn.Upsample(size=shape, mode='bilinear', align_corners=False)
    red = upsamplebyX(red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_red = bayer_images[Ellipsis, 1:2]
    green_red = torch.flip(green_red, dims=[1])  # Flip left-right
    green_red = upsamplebyX(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    green_red = torch.flip(green_red, dims=[1])  # Flip left-right
    green_red = SpaceToDepth_fact2(green_red.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_blue = bayer_images[Ellipsis, 2:3]
    green_blue = torch.flip(green_blue, dims=[0])  # Flip up-down
    green_blue = upsamplebyX(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    green_blue = torch.flip(green_blue, dims=[0])  # Flip up-down
    green_blue = SpaceToDepth_fact2(green_blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    green_at_red = (green_red[Ellipsis, 0] + green_blue[Ellipsis, 0]) / 2
    green_at_green_red = green_red[Ellipsis, 1]
    green_at_green_blue = green_blue[Ellipsis, 2]
    green_at_blue = (green_red[Ellipsis, 3] + green_blue[Ellipsis, 3]) / 2

    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = DepthToSpace_fact2(torch.stack(green_planes, dim=-1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    blue = bayer_images[Ellipsis, 3:4]
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])
    blue = upsamplebyX(blue.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    blue = torch.flip(torch.flip(blue, dims=[1]), dims=[0])

    rgb_images = torch.cat([red, green, blue], dim=-1)
    rgb_images = rgb_images.permute(0, 3, 1, 2)  # Re-Permute the tensor back to BxCxHxW format
    return rgb_images


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    if len(ccms.shape) != 3:
        ccms = ccms.squeeze()
        if len(ccms.shape) == 2:
            ccms = ccms.unsqueeze(0)
    outs = torch.matmul(images.permute(0, 2, 3, 1), ccms.permute(0, 2, 1).unsqueeze(1)).permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    # Clamps to prevent numerical instability of gradients near zero.
    outs = torch.clamp(images, min=1e-8) ** (1.0 / gamma)
    return outs


def metadata2tensor(metadata):
    xyz2cam = torch.FloatTensor(metadata['colormatrix'])
    rgb2xyz = torch.FloatTensor([[0.4124564, 0.3575761, 0.1804375],
                                 [0.2126729, 0.7151522, 0.0721750],
                                 [0.0193339, 0.1191920, 0.9503041]])
    rgb2cam = torch.mm(xyz2cam, rgb2xyz)
    rgb2cam = rgb2cam / torch.sum(rgb2cam, dim=-1, keepdim=True)
    cam2rgb = torch.inverse(rgb2cam)

    red_gain = torch.FloatTensor(metadata['red_gain'])
    blue_gain = torch.FloatTensor(metadata['blue_gain'])

    return cam2rgb.squeeze(), red_gain.squeeze().unsqueeze(0), blue_gain.squeeze().unsqueeze(0)


def raw2srgb(bayer_images, red_gains=None, blue_gains=None, cam2rgbs=None):
    """Processes a batch of Bayer RGGB images into sRGB images."""
    if red_gains is not None:
        # White balance.
        bayer_images = apply_gains_bayer(bayer_images, red_gains, blue_gains)
        # Demosaic.
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = demosaic(bayer_images)
    
    if cam2rgbs is not None:
        # Color correction.
        images = apply_ccms(images, cam2rgbs)
        # Gamma compression.
        images = torch.clamp(images, min=0.0, max=1.0)
        images = gamma_compression(images)
    return images


def rgb2srgb(linrgb, red_gains, blue_gains, cam2rgbs):
    """Processes a batch of Bayer RGGB images into sRGB images."""
    # White balance.
    rgb_wb = apply_gains_rgb(linrgb, red_gains, blue_gains)
    rgb_wb = torch.clamp(rgb_wb, min=0.0, max=1.0)
    # Color correction.
    rgb_wb_ccm = apply_ccms(rgb_wb, cam2rgbs)
    # Gamma compression.
    rgb_wb_ccm = torch.clamp(rgb_wb_ccm, min=0.0, max=1.0)
    srgb = gamma_compression(rgb_wb_ccm)
    return srgb

