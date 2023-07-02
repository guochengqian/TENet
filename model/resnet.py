import model.common as common
import torch.nn as nn
from torch.nn import Sequential as Seq

"""
resnet for different specified task
"""


class Net(nn.Module):
    def __init__(self, 
                 in_type: str = 'noisy_lr_raw',
                 mid_type: str = ['lr_raw', 'raw'], 
                 out_type: str = 'linrgb',
                 scale: int = 2,
                 output_mid: bool = False,
                 block: str = 'rrdb',
                 n_blocks: int = [1, 5, 6],
                 channels: int = 64,
                 noise_channels: int = 1,
                 norm=None,
                 act: str = 'relu',
                 **kwargs
                 ):
        super().__init__()

        bias = True if (norm is None or not norm) else False

        in_channels = 3 if 'rgb' in in_type else 4
        out_channels = 3 if 'rgb' in out_type else 4
        noise_channels = in_channels
        to_dn = 'noisy' in in_type
        self.to_upsample = False
        if 'lr' in in_type:
            self.to_upsample = True
        elif 'raw' in in_type and 'rgb' in out_type:    # demosaicking
            self.to_upsample = True
            scale = 2

        if to_dn:
            head = [common.ConvBlock(in_channels + noise_channels, channels, 3,
                                     act=act, bias=True)]
        else:
            head = [common.ConvBlock(in_channels, channels, 3,
                                     act=act, bias=True)]

        if block.lower() == 'rrdb':
            resblock = [common.RRDB(channels, channels, 3,
                                    1, bias, norm, act, 0.2)
                        for _ in range(n_blocks)]
        elif block.lower() == 'res':
            resblock = [common.ResBlock(channels, 3, norm, act, res_scale=1, bias=bias)
                        for _ in range(n_blocks)]
        else:
            raise RuntimeError('block is not supported')

        resblock += [common.ConvBlock(channels, channels, 3, bias=True, act=act)]
        self.backbone = Seq(*head, common.ShortcutBlock(Seq(*resblock)))
        self.tail = Seq(*[common.ConvBlock(channels, out_channels, 3, bias=True, act=False)])

        if self.to_upsample:
            self.sr = common.Upsampler(scale, channels, norm, act, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
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
        x = self.backbone(x)
        if self.to_upsample:
            x = self.sr(x)
        x = self.tail(x)
        return x
