import model.common as common
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

'''
ddsr_v2: bottleneck in the mid part, an end to end module.

'''
class NET(nn.Module):
    def __init__(self, n_resblock=24, n_feats=256, scale=2, bias=True, norm_type=False,
                 act_type='prelu'):
        super(NET, self).__init__()

        self.scale = scale
        m = [common.default_conv(1, n_feats, 3, stride=2)]
        m += [nn.PixelShuffle(2),
              common.ConvBlock(n_feats//4, n_feats, bias=True, act_type=act_type)
              ]

        m += [common.ResBlock(n_feats, 3, norm_type, act_type, res_scale=1, bias=bias)
                             for _ in range(n_resblock)]

        for _ in range(int(math.log(scale, 2))):
            m += [nn.PixelShuffle(2),
                  common.ConvBlock(n_feats//4, n_feats, bias=True, act_type=act_type)
                  ]

        m += [common.default_conv(n_feats, 3, 3)]

        self.model = nn.Sequential(*m)
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
        # x = self.sub_mean(x)
        return self.model(x)



