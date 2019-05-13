import model.common as common
import torch.nn as nn

"""
given bayer  -> output noise-free RGB
"""


class NET(nn.Module):
    def __init__(self, opt):
        super(NET, self).__init__()
        # parameter
        denoise = opt.denoise
        block_type = opt.block_type
        n_feats = opt.channels
        act_type = opt.act_type
        bias = opt.bias
        norm_type = opt.norm_type
        n_resblocks = opt.n_resblocks

        # architecture
        if denoise:
            dm_head = [common.ConvBlock(5, n_feats, 5,
                                        act_type=act_type, bias=True)]
        else:
            dm_head = [common.ConvBlock(4, n_feats, 5,
                                        act_type=act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            dm_resblock = [common.RRDB(n_feats, n_feats, 3,
                                       1, bias, norm_type, act_type, 0.2)
                            for _ in range(n_resblocks)]
        elif block_type.lower() == 'res':
            dm_resblock = [common.ResBlock(n_feats, 3, norm_type,
                                            act_type, res_scale=1, bias=bias)
                            for _ in range(n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')

        dm_resblock += [common.ConvBlock(n_feats, n_feats, 3, bias=True)]
        m_dm_up = [common.Upsampler(2, n_feats, norm_type, act_type, bias=bias),
                   common.ConvBlock(n_feats, 3, 3, bias=True)]

        self.model_dm = nn.Sequential(*dm_head, common.ShortcutBlock(nn.Sequential(*dm_resblock)),
                                      *m_dm_up)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model_dm(x)
        return x
    #
    # def __init__(self, n_resblock=3, n_feats=64, denoise=True, bias=True,
    #              norm_type=False, act_type='relu', block_type='rrdb'):
    #     super(NET, self).__init__()
    #
    #     if denoise:
    #         dm_head = [common.ConvBlock(5, n_feats, 5,
    #                                     act_type=act_type, bias=True)]
    #     else:
    #         dm_head = [common.ConvBlock(4, n_feats, 5,
    #                                     act_type=act_type, bias=True)]
    #     if block_type.lower() == 'rrdb':
    #         dm_resblock = [common.RRDB(n_feats, n_feats, 3,
    #                                    1, bias, norm_type, act_type, 0.2)
    #                         for _ in range(n_resblock)]
    #     elif block_type.lower() == 'res':
    #         dm_resblock = [common.ResBlock(n_feats, 3, norm_type,
    #                                         act_type, res_scale=1, bias=bias)
    #                         for _ in range(n_resblock)]
    #     else:
    #         raise RuntimeError('block_type is not supported')
    #
    #     dm_resblock += [common.ConvBlock(n_feats, n_feats, 3, bias=True)]
    #     m_dm_up = [common.Upsampler(2, n_feats, norm_type, act_type, bias=bias),
    #                common.ConvBlock(n_feats, 3, 3, bias=True)]
    #
    #     self.model_dm = nn.Sequential(*dm_head, common.ShortcutBlock(nn.Sequential(*dm_resblock)),
    #                                   *m_dm_up)
    #
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_normal_(m.weight)
    #             m.weight.requires_grad = True
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #                 m.bias.requires_grad = True
    #
    # def forward(self, x):
    #     x = self.model_dm(x)
    #     return x



