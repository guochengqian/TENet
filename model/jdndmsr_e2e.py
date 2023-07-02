import model.common as common
import torch.nn as nn
from typing import List

'''Implementation of
Deep Residual Network for Joint Demosaicing and Super-Resolution
https://arxiv.org/pdf/1802.06573.pdf
Ruofan Zhou, Radhakrishna Achanta, Sabine Susstrunk
'''

class Net(nn.Module):
    def __init__(self,
                 in_type: str = 'noisy_lr_raw',
                 mid_type: str = ['lr_raw', 'raw'],
                 out_type: str = 'linrgb',
                 scale: int = 2,
                 output_mid: bool = True,
                 channels=64,
                 n_blocks=4,  # layers of Residual Groups
                 noise_channels: int = 1,
                 **kwargs
                 ):
        super().__init__()
        self.mid_type = mid_type
        self.output_mid = output_mid

        todn = 'nois' in in_type    # needs to denoise
        todm = 'raw' in in_type     # needs to demosaic

        # noise map for each channel
        in_channels = 4 if todm else 3
        out_channels = 4 if 'raw' in out_type else 3
        noisy_channels = noise_channels if todn else 0
        self.stem = nn.Sequential(common.ConvBlock(in_channels + noisy_channels, channels, 3,
                                  act='relu', bias=True)
                                  )

        # for backbone
        self.mid_type = mid_type
        # backbone
        self.stage1 = nn.Sequential(
            *[common.RG(channels) for _ in range(n_blocks//2)])
        stage2 = [common.RG(channels) for _ in range(
            n_blocks//2)] + [common.ConvBlock(channels, channels, 3, act='relu', bias=True)]
        self.stage2 = nn.Sequential(*stage2)
        self.out_branch = nn.Sequential(nn.ConvTranspose2d(channels, channels, 2*scale, 2*scale, 0),
                                        nn.ReLU(True),  
                                        common.ConvBlock(channels, out_channels, act=False, bias=True))
        if mid_type is not None:
            # only support DM.
            self.mid_branch = common.ConvBlock(
                channels, 4, act=False, bias=True)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.data.zero_()
                m.bias.requires_grad = True
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.data.zero_()
                m.bias.requires_grad = True

    def forward(self, x):
        out_stem = self.stem(x)
        out_stage1 = self.stage1(out_stem)
        out_stage2 = self.stage2(out_stage1) + out_stem
        out = self.out_branch(out_stage2)
        return out