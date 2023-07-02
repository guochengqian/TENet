import model.common as common
from model import attention
import torch
import torch.nn as nn
from collections import OrderedDict

'''Implementation of TENet
Rethinking the pipeline
'''


class Net(nn.Module):
    def __init__(self,
                 in_type: str = 'noisy_lr_raw',
                 mid_type: str = ['lr_raw', 'raw'], 
                 out_type: str = 'linrgb',
                 scale: int = 2,
                 output_mid: bool = False,
                 block: str = 'rrdb',
                 n_blocks: int = 12,
                 channels: int = 64,
                 noise_channels: int = 1,
                 norm=None,
                 act: str = 'relu',
                 **kwargs):
        super(Net, self).__init__()
        bias = True if (norm is None or not norm) else False
        denoise = 'noisy' in in_type
        n_blocks = n_blocks // 2
        self.mid_type = mid_type
        self.output_mid = output_mid
        if self.mid_type is None or ('raw' in self.mid_type and 'lr' not in self.mid_type):  # Default, DN -> SR -> DM
            scale1 = scale
            scale2 = 2
            channel1 = 4
        elif 'lr_raw' in self.mid_type:
            scale1 = 1
            scale2 = 2*scale
            channel1 = 4
        else:  # DN -> DM -> SR
            scale1 = 2
            scale2 = scale
            channel1 = 3

        # input sensor raw bayer image with noise, shape: (1 x H x W).
        # to process each channel differently (R, G, G, B),
        #   we reshape the bayer into a (4-channels image map) with shape (4 x H/2 x W/2).
        # Then, if denoise, TENet requires a noise map as input (the sigma value of Gaussian distribution).
        #   Thus, the input shape: shape (5 x H/2 x W/2).

        # First step, joint Denoising + Super-resolution, output shape: (4 x H x W).
        if denoise:
            m_head = [common.ConvBlock(4 + noise_channels, channels, 3,
                                       act=None, bias=True)]
        else:
            m_head = [common.ConvBlock(4, channels, 3,
                                       act=None, bias=True)]

        if block.lower() == 'rrdb':
            m_resblock1 = [common.RRDB(channels, channels, 3,
                                       1, bias, norm, act, 0.2)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.RRDB(channels, channels, 3,
                                       1, bias, norm, act, 0.2)
                           for _ in range(n_blocks)]
        elif block.lower() == 'dudb':
            m_resblock1 = [common.DUDB(channels, 3, 1, bias,
                                       norm, act, 0.2)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.DUDB(channels, 3, 1, bias,
                                       norm, act, 0.2)
                           for _ in range(n_blocks)]
        elif block.lower() == 'res':
            m_resblock1 = [common.ResBlock(channels, 3, norm,
                                           act, res_scale=1, bias=bias)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.ResBlock(channels, 3, norm,
                                           act, res_scale=1, bias=bias)
                           for _ in range(n_blocks)]

        elif block.lower() == 'eam':
            m_resblock1 = [common.EAMBlock(channels, channels)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.EAMBlock(channels, channels)
                           for _ in range(n_blocks)]

        elif block.lower() == 'drlm':
            m_resblock1 = [common.DRLM(channels, channels)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.DRLM(channels, channels)
                           for _ in range(n_blocks)]

        elif block.lower() == 'rrg':
            m_resblock1 = [common.RRG(channels)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.RRG(channels)
                           for _ in range(n_blocks)]

        elif block.lower() == 'rcab':
            m_resblock1 = [common.RCABGroup(channels)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.RCABGroup(channels)
                           for _ in range(n_blocks)]

        elif block.lower() == 'nlsa':
            m_resblock1 = [common.ResBlock(channels, 3, norm=norm,
                                           act=act, bias=bias)
                           for _ in range(n_blocks)]
            m_resblock1.append(attention.NonLocalSparseAttention(channels=channels))
            m_resblock2 = [common.ResBlock(channels, 3, norm=norm,
                                           act=act, bias=bias)
                           for _ in range(n_blocks)]
            m_resblock2.append(attention.NonLocalSparseAttention(channels=channels))
        else:
            raise RuntimeError('block :{} is not supported'.format(block))

        # super-resolution module (using pixelshuffle layer). output: (4 x H x W)
        m_resblock1_up = [common.Upsampler(scale1, channels, norm, act, bias=bias)]
        m_resblock2_up = [common.Upsampler(scale2, channels, norm, act, bias=bias)]

        # branch for sr_raw output. From 4-channels R-G-G-B images with shape (4 x H x W)
        # to sensor raw bayer image(1x 4H x 4W)
        m_branch1 = [common.ConvBlock(channels, channel1, 3, act=False, bias=True)]
        m_branch2 = [common.ConvBlock(channels, 3, 3, act=False, bias=True)]

        self.stage1 = nn.Sequential(*m_head, common.ShortcutBlock(nn.Sequential(*m_resblock1)), *m_resblock1_up)
        self.mid_branch = nn.Sequential(*m_branch1)
        self.stage2 = nn.Sequential(common.ShortcutBlock(nn.Sequential(*m_resblock2)), *m_resblock2_up, *m_branch2)
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


if __name__ == '__main__':
    import argparse
    from TorchTools.ArgsTools.base_args import BaseArgs
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args
    model = NET(args)
    model.load_state_dict_from_other_pipeline(args.pretrain_other)
    print(model)

