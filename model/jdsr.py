import model.common as common
import torch.nn as nn
import math

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
                 n_blocks=4,  # layers of Residual Groups
                 noise_channels: int = 1,
                 original=False, 
                 **kwargs
                 ):
        super().__init__()
        channels = 256
        #channels = 64
        act = 'prelu'
        bias = True
        self.mid_type = mid_type
        self.output_mid = output_mid
        if mid_type is None or ('raw' in mid_type and 'lr' not in mid_type):  # Default, DN -> SR -> DM
            scale1 = scale
            scale2 = 2
            midout_channel = 4
        elif 'lr_raw' in mid_type:
            scale1 = 1
            scale2 = 2*scale
            midout_channel = 4
        else:  # DN -> DM -> SR
            scale1 = 2
            scale2 = scale
            midout_channel = 3

        if 'noisy' in in_type:
            if 'raw' in in_type:
                noise_channel = 4
            else:
                noise_channel = 3
        else:
            noise_channel = 0
        in_channel = 4

        midout_channel = midout_channel
        m_stem = [common.default_conv(in_channel+noise_channel, channels, 3)]

        m_block1 = [common.ResBlock(channels, 3, act=act, res_scale=1, bias=bias, last_act=True) for _ in range(n_blocks//2)]
        m_block2 = [common.ResBlock(channels, 3, act=act, res_scale=1, bias=bias, last_act=True) for _ in range(n_blocks//2)]

        m_block1_up = [common.UpsamplerSmall(scale2, channels, act=act, bias=bias)]
        m_block2_up = [common.UpsamplerSmall(scale2, channels, act=act, bias=bias)]
        m_branch2 = [common.ConvBlock(channels, 3, 3, act=False, bias=True)]

        if not original and mid_type is None:
            self.stage1 = nn.Sequential(*m_stem, *m_block1)
        else:
            self.stage1 = nn.Sequential(*m_stem, m_block1_up, *m_block1)
        self.mid_branch = common.ConvBlock(channels, midout_channel, 3, act=False, bias=True) if mid_type is not None else None 
        
        if not original and mid_type is None:
            self.stage2 = nn.Sequential(*m_block2, *m_block1_up, *m_block2_up, *m_branch2)
        else:    
            self.stage2 = nn.Sequential(*m_block2, *m_block2_up, *m_branch2)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
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
        out_stage1 = self.stage1(x)
        if (self.mid_type is not None and self.training) or self.output_mid:
            mid_out = self.mid_branch(out_stage1)
        out_stage2 = self.stage2(out_stage1)

        if (self.mid_type is not None and self.training) or self.output_mid:
            return mid_out, out_stage2
        else:
            return out_stage2