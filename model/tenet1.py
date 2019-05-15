import model.common as common
import torch.nn as nn


"""
given LR bayer  -> output HR noise-free RGB
"""
class NET(nn.Module):
    def __init__(self, opt):
        super(NET, self).__init__()

        sr_n_resblocks = opt.sr_n_resblocks
        dm_n_resblocks = opt.dm_n_resblocks
        sr_n_feats = opt.channels
        dm_n_feats = opt.channels
        scale = opt.scale

        denoise = opt.denoise
        block_type = opt.block_type
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type

        # define sr module
        if denoise:
            m_sr_head = [common.ConvBlock(5, sr_n_feats, 5,
                                          act_type=act_type, bias=True)]
        else:
            m_sr_head = [common.ConvBlock(4, sr_n_feats, 5,
                                          act_type=act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            m_sr_resblock = [common.RRDB(sr_n_feats, sr_n_feats, 3,
                                         1, bias, norm_type, act_type, 0.2)
                             for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_sr_resblock = [common.DUDB(sr_n_feats, 3, 1, bias,
                                         norm_type, act_type, 0.2)
                             for _ in range(sr_n_resblocks)]
        elif block_type.lower() == 'res':
            m_sr_resblock = [common.ResBlock(sr_n_feats, 3, norm_type,
                                             act_type, res_scale=1, bias=bias)
                             for _ in range(sr_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')

        m_sr_resblock += [common.ConvBlock(sr_n_feats, sr_n_feats, 3, bias=bias)]
        m_sr_up = [common.Upsampler(scale, sr_n_feats, norm_type, act_type, bias=bias),
                   common.ConvBlock(sr_n_feats, sr_n_feats, 3, bias=True)]

        # define demosaick module
        m_dm_head = [common.ConvBlock(sr_n_feats, dm_n_feats, 5,
                                      act_type=act_type, bias=True)]

        if block_type.lower() == 'rrdb':
            m_dm_resblock = [common.RRDB(dm_n_feats, dm_n_feats, 3,
                                         1, bias, norm_type, act_type, 0.2)
                             for _ in range(dm_n_resblocks)]
        elif block_type.lower() == 'dudb':
            m_dm_resblock = [common.DUDB(dm_n_feats, 3, 1, bias,
                                         norm_type, act_type, 0.2)
                             for _ in range(dm_n_resblocks)]
        elif block_type.lower() == 'res':
            m_dm_resblock = [common.ResBlock(dm_n_feats, 3, norm_type,
                                             act_type, res_scale=1, bias=bias)
                             for _ in range(dm_n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')

        m_dm_resblock += [common.ConvBlock(dm_n_feats, dm_n_feats, 3, bias=bias)]
        m_dm_up = [common.Upsampler(2, dm_n_feats, norm_type, act_type, bias=bias),
                   common.ConvBlock(dm_n_feats, 3, 3, bias=True)]

        self.model_sr = nn.Sequential(*m_sr_head, common.ShortcutBlock(nn.Sequential(*m_sr_resblock)),
                                      *m_sr_up)
        self.model_dm = nn.Sequential(*m_dm_head, common.ShortcutBlock(nn.Sequential(*m_dm_resblock)),
                                      *m_dm_up)

        for m in self.modules():
            # pdb.set_trace()
            if isinstance(m, nn.Conv2d):
                # Xavier
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        return self.model_dm(self.model_sr(x))



