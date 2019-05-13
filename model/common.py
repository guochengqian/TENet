import math

import torch
import torch.nn as nn

##############################
#    Basic layer
##############################
def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm_layer(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


def default_conv(in_channelss, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size//2), stride=stride, bias=bias)


class ConvBlock(nn.Sequential):
    def __init__(
        self, in_channelss, out_channels, kernel_size=3, stride=1, bias=False,
            norm_type=False, act_type='relu'):

        m = [default_conv(in_channelss, out_channels, kernel_size, stride=stride, bias=bias)]
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, out_channels) if norm_type else None
        if norm:
            m.append(norm)
        if act is not None:
            m.append(act)
        super(ConvBlock, self).__init__(*m)


##############################
#    Useful Blocks
##############################
class ShortcutBlock(nn.Module):
    #Elementwise sum the output of a submodule to its input
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        tmpstr = 'Identity + \n|'
        modstr = self.sub.__repr__().replace('\n', '\n|')
        tmpstr = tmpstr + modstr
        return tmpstr


class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size=3,
            norm_type=False, act_type='relu', bias=False, res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, n_feats) if norm_type else None
        for i in range(2):
            m.append(default_conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm:
                m.append(norm)
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualDenseBlock5(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR18)
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(ResidualDenseBlock5, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv2 = ConvBlock(nc+gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv3 = ConvBlock(nc+2*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv4 = ConvBlock(nc+3*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv5 = ConvBlock(nc+4*gc, gc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(self.res_scale) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, nc, gc=32, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(RRDB, self).__init__()
        self.res_scale = res_scale
        self.RDB1 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)
        self.RDB2 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)
        self.RDB3 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm_type, act_type, res_scale)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(self.res_scale) + x


class SkipUpDownBlock(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR18)
    """

    def __init__(self, nc, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(SkipUpDownBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, nc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.conv2 = ConvBlock(2*nc, 2*nc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)
        self.up = nn.PixelShuffle(2)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(nc, nc, kernel_size, stride, bias=bias, norm_type=norm_type,
                               act_type=act_type)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.up(torch.cat((x, x1, x2), 1))
        x3 = self.conv3(self.pool(x3))
        return x3.mul(self.res_scale) + x


class DUDB(nn.Module):
    """
    Dense Up Down Block
    """

    def __init__(self, nc, kernel_size=3, stride=1, bias=True,
                 norm_type=None, act_type='leakyrelu', res_scale=0.2):
        super(DUDB, self).__init__()
        self.res_scale = res_scale
        self.UDB1 = SkipUpDownBlock(nc, kernel_size, stride, bias,
                                    norm_type, act_type, res_scale)
        self.UDB2 = SkipUpDownBlock(nc, kernel_size, stride, bias,
                                    norm_type, act_type, res_scale)
        self.UDB3 = SkipUpDownBlock(nc, kernel_size, stride, bias,
                                    norm_type, act_type, res_scale)

    def forward(self, x):
        return self.UDB3(self.UDB2(self.UDB1(x))).mul(self.res_scale) + x


###########################
#  Upsamler layer
##########################
class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, norm_type=False, act_type='relu', bias=False):

        m = []
        act = act_layer(act_type) if act_type else None
        norm = norm_layer(norm_type, n_feats) if norm_type else None
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(default_conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if norm: m.append(norm)
                if act is not None: m.append(act)

        elif scale == 3:
            m.append(default_conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if norm: m.append(norm)
            if act is not None: m.append(act)
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DownsamplingShuffle(nn.Module):

    def __init__(self, scale):
        super(DownsamplingShuffle, self).__init__()
        self.scale = scale

    def forward(self, input):
        """
        input should be 4D tensor N, C, H, W
        :return: N, C*scale**2,H//scale,W//scale
        """
        N, C, H, W = input.size()
        assert H % self.scale == 0, 'Please Check input and scale'
        assert W % self.scale == 0, 'Please Check input and scale'
        map_channels = self.scale ** 2
        channels = C * map_channels
        out_height = H // self.scale
        out_width = W // self.scale

        input_view = input.contiguous().view(
            N, C, out_height, self.scale, out_width, self.scale)

        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

        return shuffle_out.view(N, channels, out_height, out_width)


def demosaick_layer(input):
    demo = nn.PixelShuffle(2)
    return demo(input)


#############################
#  counting number
#
#############################
def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
