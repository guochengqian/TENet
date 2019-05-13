import model.common as common
import torch.nn as nn

"""
given bayer  -> output noise-free bayer
"""


class NET(nn.Module):
    def __init__(self, opt):
        super(NET, self).__init__()

        n_resblocks = opt.n_resblocks
        n_feats = opt.channels
        bias = opt.bias
        norm_type = opt.norm_type
        act_type = opt.act_type
        block_type = opt.block_type

        head = [common.ConvBlock(5, n_feats, 5, act_type=act_type, bias=True)]
        if block_type.lower() == 'rrdb':
            resblock = [common.RRDB(n_feats, n_feats, 3,
                                       1, bias, norm_type, act_type, 0.2)
                            for _ in range(n_resblocks)]
        elif block_type.lower() == 'res':
            resblock = [common.ResBlock(n_feats, 3, norm_type, act_type, res_scale=1, bias=bias)
                            for _ in range(n_resblocks)]
        else:
            raise RuntimeError('block_type is not supported')

        resblock += [common.ConvBlock(n_feats, n_feats, 3, bias=True)]
        tail = [common.Upsampler(2, n_feats, norm_type, act_type, bias=bias),
                   common.ConvBlock(n_feats, 1, 3, bias=True)]

        self.model = nn.Sequential(*head, common.ShortcutBlock(nn.Sequential(*resblock)), *tail)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


