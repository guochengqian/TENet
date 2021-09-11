import model.common as common
from model import attention
import torch
import torch.nn as nn
from collections import OrderedDict

'''Implementation of TENet
Rethinking the pipeline
'''


class NET(nn.Module):
    def __init__(self, opt):
        super(NET, self).__init__()
        assert 'raw' in opt.in_type and 'rgb' in opt.out_type

        n_blocks = opt.n_blocks
        n_feats = opt.channels

        denoise = 'noisy' in opt.in_type
        block_type = opt.block_type
        norm_type = opt.norm_type
        act_type = opt.act_type
        bias = opt.bias

        self.mid_type = opt.mid_type
        self.output_mid = opt.output_mid
        if self.mid_type is None or ('raw' in self.mid_type and 'lr' not in self.mid_type):  # Default, DN -> SR -> DM
            scale1 = opt.scale
            scale2 = 2
            channel1 = 4
        elif 'lr_raw' in self.mid_type:
            scale1 = 1
            scale2 = 2*opt.scale
            channel1 = 4
        else:  # DN -> DM -> SR
            scale1 = 2
            scale2 = opt.scale
            channel1 = 3

        # input sensor raw bayer image with noise, shape: (1 x H x W).
        # to process each channel differently (R, G, G, B),
        #   we reshape the bayer into a (4-channels image map) with shape (4 x H/2 x W/2).
        # Then, if denoise, TENet requires a noise map as input (the sigma value of Gaussian distribution).
        #   Thus, the input shape: shape (5 x H/2 x W/2).

        # First step, joint Denoising + Super-resolution, output shape: (4 x H x W).
        noisy_channel = 4 if 'raw' in opt.in_type else 3
        if denoise:
            m_head = [common.ConvBlock(4 + noisy_channel, n_feats, 3,
                                       act_type=None, bias=True)]
        else:
            m_head = [common.ConvBlock(4, n_feats, 3,
                                       act_type=None, bias=True)]

        if block_type.lower() == 'rrdb':
            m_resblock1 = [common.RRDB(n_feats, n_feats, 3,
                                       1, bias, norm_type, act_type, 0.2)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.RRDB(n_feats, n_feats, 3,
                                       1, bias, norm_type, act_type, 0.2)
                           for _ in range(n_blocks)]
        elif block_type.lower() == 'dudb':
            m_resblock1 = [common.DUDB(n_feats, 3, 1, bias,
                                       norm_type, act_type, 0.2)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.DUDB(n_feats, 3, 1, bias,
                                       norm_type, act_type, 0.2)
                           for _ in range(n_blocks)]
        elif block_type.lower() == 'res':
            m_resblock1 = [common.ResBlock(n_feats, 3, norm_type,
                                           act_type, res_scale=1, bias=bias)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.ResBlock(n_feats, 3, norm_type,
                                           act_type, res_scale=1, bias=bias)
                           for _ in range(n_blocks)]

        elif block_type.lower() == 'eam':
            m_resblock1 = [common.EAMBlock(n_feats, n_feats)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.EAMBlock(n_feats, n_feats)
                           for _ in range(n_blocks)]

        elif block_type.lower() == 'drlm':
            m_resblock1 = [common.DRLM(n_feats, n_feats)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.DRLM(n_feats, n_feats)
                           for _ in range(n_blocks)]

        elif block_type.lower() == 'rrg':
            m_resblock1 = [common.RRG(n_feats)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.RRG(n_feats)
                           for _ in range(n_blocks)]

        elif block_type.lower() == 'rcab':
            m_resblock1 = [common.RCABGroup(n_feats)
                           for _ in range(n_blocks)]
            m_resblock2 = [common.RCABGroup(n_feats)
                           for _ in range(n_blocks)]

        elif block_type.lower() == 'nlsa':
            m_resblock1 = [common.ResBlock(n_feats, 3, norm_type,
                                           act_type, res_scale=1, bias=bias)
                           for _ in range(n_blocks)]
            m_resblock1.append(attention.NonLocalSparseAttention(channels=n_feats))
            m_resblock2 = [common.ResBlock(n_feats, 3, norm_type,
                                           act_type, res_scale=1, bias=bias)
                           for _ in range(n_blocks)]
            m_resblock2.append(attention.NonLocalSparseAttention(channels=n_feats))
        else:
            raise RuntimeError('block_type :{} is not supported'.format(block_type))

        # super-resolution module (using pixelshuffle layer). output: (4 x H x W)
        m_resblock1_up = [common.Upsampler(scale1, n_feats, norm_type, act_type, bias=bias)]
        m_resblock2_up = [common.Upsampler(scale2, n_feats, norm_type, act_type, bias=bias)]

        # branch for sr_raw output. From 4-channels R-G-G-B images with shape (4 x H x W)
        # to sensor raw bayer image(1x 4H x 4W)
        m_branch1 = [common.ConvBlock(n_feats, channel1, 3, act_type=False, bias=True)]
        m_branch2 = [common.ConvBlock(n_feats, 3, 3, act_type=False, bias=True)]

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

    def load_state_dict_from_other_pipeline(self, pretrain_other):
        ckpt_other = torch.load(pretrain_other)['state_dict']
        model_dict = self.state_dict()

        # rename ckpt (avoid name is not same because of multi-gpus)
        is_model_multi_gpus = True if list(model_dict)[0].split('.')[0] == 'module' else False
        is_ckpt_multi_gpus = True if list(ckpt_other)[0].split('.')[0] == 'module' else False

        if not (is_model_multi_gpus == is_ckpt_multi_gpus):
            temp_dict = OrderedDict()
            for k, v in ckpt_other.items():
                if is_ckpt_multi_gpus:
                    name = k[7:]  # remove 'module.'
                else:
                    name = 'module.' + k  # add 'module'
                temp_dict[name] = v
            ckpt_other = temp_dict

        # 1. filter out unnecessary keys
        pretrained_state = {k: v for k, v in ckpt_other.items() if
                            k in model_dict and v.size() == model_dict[k].size()}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_state)
        self.load_state_dict(model_dict)

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

