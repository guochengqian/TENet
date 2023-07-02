'''Implementation of TENet v0
the previous version of TENet without the detachable layer design
'''
# import __init__
import torch
from torch import nn
import model.common as common
from model import attention
from typing import List, Tuple


class Net(nn.Module):
    types = ['noisy', 'lr', 'raw', 'linrgb', 'rgb']
    def __init__(self,
                 in_type: str = 'noisy_lr_raw',
                 mid_type: str = ['lr_raw', 'raw'], 
                 out_type: str = 'linrgb',
                 scale: int = 2,
                 output_mid: bool = True,
                 block: str = 'rrdb',
                 n_blocks: int = [1, 5, 6],
                 channels: int = 64,
                 noise_channels: int = 1,
                 norm=None,
                 act: str = 'relu',
                 **kwargs
                 ):
        super().__init__()
        assert 'raw' in in_type and 'rgb' in out_type
        bias = True if (norm is None or not norm) else False
        self.output_mid = output_mid
        todn = 'nois' in in_type    # needs to denoise
        todm = 'raw' in in_type     # needs to demosaic
        tosr = 'lr' in in_type      # needs to super-resolve
        
        # noise map for each channel
        in_channels = 4 if todm else 3
        noisy_channels = noise_channels if todn else 0
        m_head = common.ConvBlock(in_channels + noisy_channels, channels, 3,
                                  act=None, bias=True) 
        
        # for backbone
        self.mid_type = mid_type
        mid_types = mid_type + [out_type] if mid_type is not None else [out_type]
        self.n_stages = len(mid_types)
        
        if isinstance(n_blocks, List):
            blocks_idx = [0] + [sum(n_blocks[:i+1]) for i, v in enumerate(n_blocks)]
            n_blocks = sum(n_blocks)
        else:
            blocks_idx = [0, n_blocks // self.n_stages, (n_blocks * 2) // self.n_stages, n_blocks]

        scales = [] # the upsampling scale ratios for each module
        denoise = [] # whether denoise happens at each module
        branch_channels = [] # output channels of each brach
        for mid_type in mid_types:
            scale_c = 1
            todn_c = 'nois' in mid_type
            todm_c = 'raw' in mid_type
            tosr_c = 'lr' in mid_type

            if todn and not todn_c:
                denoise.append(True)
                todn = False
            else:
                denoise.append(False)
            
            if todm and not todm_c:
                scale_c *= 2
                todm = False
                
            if tosr and not tosr_c:
                scale_c *= scale           
                tosr = False
                
            scales.append(scale_c)
            if todm_c:
                branch_channels.append(4)
            else:
                branch_channels.append(3)

        if block.lower() == 'rrdb':
            blocks = [common.RRDB(channels, channels, 3,
                                1, bias, norm, act, 0.2)
                    for _ in range(n_blocks)]

        elif block.lower() == 'dudb':
            blocks = [common.DUDB(channels, 3, 1, bias,
                                norm, act, 0.2)
                    for _ in range(n_blocks)]

        elif block.lower() == 'res':
            blocks = [common.ResBlock(channels, 3, bias, norm,
                                    act, res_scale=1)
                    for _ in range(n_blocks)]

        elif block.lower() == 'eam':
            blocks = [common.EAMBlock(channels, channels)
                    for _ in range(n_blocks)]
        elif block.lower() == 'drlm':
            blocks = [common.DRLM(channels, channels)
                    for _ in range(n_blocks)]

        elif block.lower() == 'rrg':
            blocks = [common.RRG(channels)
                    for _ in range(n_blocks)]
        elif block.lower() == 'rcab':
            blocks = [common.RCABGroup(channels)
                    for _ in range(n_blocks)]

        elif block.lower() == 'nlsa':
            for i in range(self.n_stages):
                blocks.append(*[common.ResBlock(channels, 3, norm,
                                        act, res_scale=1, bias=bias)
                        for _ in range(n_blocks)])
                blocks.append(
                    attention.NonLocalSparseAttention(channels=channels))
        else:
            raise RuntimeError('block :{} is not supported'.format(block))

        # backbone
        self.stages = nn.ModuleList([]) 
        self.branches = nn.ModuleList([])
        for i in range(self.n_stages):
            stage = [m_head] if m_head is not None else []  
            stage.append(common.ShortcutBlock(nn.Sequential(*blocks[blocks_idx[i]:blocks_idx[i+1]])))
            if scales[i] > 1:
                stage.append(common.Upsampler(scales[i], channels, norm, act, bias=bias))
            m_head = None
            self.stages.append(nn.Sequential(*stage))
            self.branches.append(common.ConvBlock(channels, branch_channels[i], act=False, bias=True))
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
        out = []
        for i in range(self.n_stages):
            x = self.stages[i](x)
            if self.output_mid or (self.mid_type is not None and self.training):
                out.append(self.branches[i](x))
            elif i == self.n_stages - 1:
                x = self.branches[i](x) 
        if self.output_mid or (self.mid_type is not None and self.training):
            return out 
        else:
            return x


if __name__ == "__main__":
    model = Net()
    x = torch.randn(4, 8, 100, 100)
    model(x)