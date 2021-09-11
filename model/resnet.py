import model.common as common
import torch.nn as nn
from torch.nn import Sequential as Seq

"""
resnet for different specified task
"""


class NET(nn.Module):
    def __init__(self, opt):
        super(NET, self).__init__()

        n_blocks = opt.n_blocks
        n_feats = opt.channels

        block_type = opt.block_type
        norm_type = opt.norm_type
        act_type = opt.act_type
        bias = opt.bias

        in_channels = 3 if 'rgb' in opt.in_type else 4
        out_channels = 3 if 'rgb' in opt.out_type else 4
        noise_channels = in_channels
        use_denoise = 'noisy' in opt.in_type
        self.use_sr = False
        if 'lr' in opt.in_type:
            self.use_sr = True
            scale = opt.scale
        elif 'raw' in opt.in_type and 'rgb' in opt.out_type:    # demosaicking
            self.use_sr = True
            scale = 2

        if use_denoise:
            head = [common.ConvBlock(in_channels + noise_channels, n_feats, 3,
                                     act_type=act_type, bias=True)]
        else:
            head = [common.ConvBlock(in_channels, n_feats, 3,
                                     act_type=act_type, bias=True)]

        if block_type.lower() == 'rrdb':
            resblock = [common.RRDB(n_feats, n_feats, 3,
                                    1, bias, norm_type, act_type, 0.2)
                        for _ in range(n_blocks)]
        elif block_type.lower() == 'res':
            resblock = [common.ResBlock(n_feats, 3, norm_type, act_type, res_scale=1, bias=bias)
                        for _ in range(n_blocks)]
        else:
            raise RuntimeError('block_type is not supported')

        resblock += [common.ConvBlock(n_feats, n_feats, 3, bias=True, act_type=act_type)]

        self.backbone = Seq(*head, common.ShortcutBlock(Seq(*resblock)))
        self.tail = Seq(*[common.ConvBlock(n_feats, out_channels, 3, bias=True, act_type=False)])

        if self.use_sr:
            self.sr = common.Upsampler(scale, n_feats, norm_type, act_type, bias=bias)

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
        if self.use_sr:
            x = self.sr(x)
        x = self.tail(x)
        return x
