import math
import numpy as np
import torch
import torch.nn as nn
from .antialias import Downsample as downsamp


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


##############################
#    Basic layer
##############################
def act_layer(act, inplace=False, neg_slope=0.25, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'sigmoid':
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


def norm_layer(norm, nc):
    # helper selecting normalization layer
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer


def default_conv(in_channelss, out_channels, kernel_size, stride=1, bias=False):
    return nn.Conv2d(
        in_channelss, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class ConvBlock(nn.Sequential):
    def __init__(
            self, in_channelss, out_channels, kernel_size=3, stride=1, bias=False,
            norm=False, act='relu'):

        m = [default_conv(in_channelss, out_channels, kernel_size, stride=stride, bias=bias)]
        act = act_layer(act) if act else None
        norm = norm_layer(norm, out_channels) if norm else None
        if norm:
            m.append(norm)
        if act is not None:
            m.append(act)
        super(ConvBlock, self).__init__(*m)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range=1.0, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1., 1., 1.), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


##############################
#    Useful Blocks
##############################
class ShortcutBlock(nn.Module):
    # Elementwise sum the output of a submodule to its input
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
            bias=False, norm=False, act='relu', res_scale=1,
            layers=2, last_act=False):
        super(ResBlock, self).__init__()
        m = []
        act = act_layer(act) if act else None
        norm = norm_layer(norm, n_feats) if norm else None
        for i in range(layers):
            m.append(default_conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm:
                m.append(norm)
            if i != layers - 1 or last_act:
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
                 norm=None, act='leakyrelu', res_scale=0.2):
        super(ResidualDenseBlock5, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, gc, kernel_size, stride, bias=bias, norm=norm,
                               act=act)
        self.conv2 = ConvBlock(nc + gc, gc, kernel_size, stride, bias=bias, norm=norm,
                               act=act)
        self.conv3 = ConvBlock(nc + 2 * gc, gc, kernel_size, stride, bias=bias, norm=norm,
                               act=act)
        self.conv4 = ConvBlock(nc + 3 * gc, gc, kernel_size, stride, bias=bias, norm=norm,
                               act=act)
        self.conv5 = ConvBlock(nc + 4 * gc, gc, kernel_size, stride, bias=bias, norm=norm,
                               act=act)

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
                 norm=None, act='leakyrelu', res_scale=0.2):
        super(RRDB, self).__init__()
        self.res_scale = res_scale
        self.RDB1 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm, act, res_scale)
        self.RDB2 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm, act, res_scale)
        self.RDB3 = ResidualDenseBlock5(nc, gc, kernel_size, stride, bias,
                                        norm, act, res_scale)

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
                 norm=None, act='leakyrelu', res_scale=0.2):
        super(SkipUpDownBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.res_scale = res_scale
        self.conv1 = ConvBlock(nc, nc, kernel_size, stride, bias=bias, norm=norm,
                               act=act)
        self.conv2 = ConvBlock(2 * nc, 2 * nc, kernel_size, stride, bias=bias, norm=norm,
                               act=act)
        self.up = nn.PixelShuffle(2)
        self.pool = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(nc, nc, kernel_size, stride, bias=bias, norm=norm,
                               act=act)

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
                 norm=None, act='leakyrelu', res_scale=0.2):
        super(DUDB, self).__init__()
        self.res_scale = res_scale
        self.UDB1 = SkipUpDownBlock(nc, kernel_size, stride, bias,
                                    norm, act, res_scale)
        self.UDB2 = SkipUpDownBlock(nc, kernel_size, stride, bias,
                                    norm, act, res_scale)
        self.UDB3 = SkipUpDownBlock(nc, kernel_size, stride, bias,
                                    norm, act, res_scale)

    def forward(self, x):
        return self.UDB3(self.UDB2(self.UDB1(x))).mul(self.res_scale) + x


# --------------------- EAM Block ----------------------------------
# from paper RIDNet: https://github.com/saeed-anwar/RIDNet
# Real Image Denoising with Feature Attention: https://arxiv.org/abs/1904.07396
class MergeRunDual(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(MergeRunDual, self).__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, 3, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, ksize, stride, 4, 4),
            nn.ReLU(inplace=True)
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.body1(x)
        out2 = self.body2(x)
        c = torch.cat([out1, out2], dim=1)
        c_out = self.body3(c)
        out = c_out + x
        return out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = ConvBlock(channel, channel // reduction, 1, 1)
        self.c2 = ConvBlock(channel // reduction, channel, 1, 1, act='sigmoid')

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.c1(y)
        y2 = self.c2(y1)
        return x * y2


class EAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(EAMBlock, self).__init__()

        self.r1 = MergeRunDual(in_channels, out_channels)
        self.r2 = ResBlock(out_channels)
        self.r3 = ResBlock(out_channels, layers=3)
        self.ca = CALayer(out_channels)

    def forward(self, x):
        r1 = self.r1(x)
        r2 = self.r2(r1)
        r3 = self.r3(r2)
        out = self.ca(r3)
        return out


# --------------------- End of EAM Block ----------------------------------


# ------------------------------- DRLM ------------------------------------
#  dense residual Laplacian module (DRLM) proposed in Densely Residual Laplacian Super-Resolution
# https://arxiv.org/pdf/1906.12021.pdf
class DRLM(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(DRLM, self).__init__()
        self.r1 = ResBlock(in_channels)
        self.r2 = ResBlock(in_channels * 2)
        self.r3 = ResBlock(in_channels * 4)
        self.g = ConvBlock(in_channels * 8, out_channels, 1, 1)
        self.ca = CALayer(out_channels)

    def forward(self, x):
        c0 = x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)

        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)

        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out


# -------------------------- End of DRLM ------------------------------------


# ---------- Recursive Residual Group (RRG) ----------
# From paper Learning Enriched Features for Real Image Restoration and Enhancement
# MIRNet: https://github.com/swz30/MIRNet
# https://arxiv.org/abs/2003.06792

# ---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = ConvBlock(2, 1, kernel_size, act=None)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


# ---------- Dual Attention Unit (DAU) ----------
class DAU(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.PReLU(), res_scale=1):
        super(DAU, self).__init__()
        modules_body = [default_conv(n_feat, n_feat, kernel_size, bias=bias), act,
                        default_conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        # Spatial Attention
        self.SA = spatial_attn_layer()

        # Channel Attention
        self.CA = CALayer(n_feat, reduction)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


# ---------- Resizing Modules ----------
class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                 nn.PReLU(),
                                 downsamp(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(downsamp(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels))
            in_channels = int(in_channels * stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3,
                                                    stride=2, padding=1, output_padding=1, bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


# ---------- Multi-Scale Residual Block (MSRB) ----------
class MSRB(nn.Module):
    def __init__(self, n_feat, height, width, stride, bias):
        super(MSRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width
        self.blocks = nn.ModuleList([nn.ModuleList([DAU(int(n_feat * stride ** i))] * width) for i in range(height)])

        INDEX = np.arange(0, width, 2)
        FEATS = [int((stride ** i) * n_feat) for i in range(height)]
        SCALE = [2 ** i for i in range(1, height)]

        self.last_up = nn.ModuleDict()
        for i in range(1, height):
            self.last_up.update({f'{i}': UpSample(int(n_feat * stride ** i), 2 ** i, stride)})

        self.down = nn.ModuleDict()
        self.up = nn.ModuleDict()

        i = 0
        SCALE.reverse()
        for feat in FEATS:
            for scale in SCALE[i:]:
                self.down.update({f'{feat}_{scale}': DownSample(feat, scale, stride)})
            i += 1

        i = 0
        FEATS.reverse()
        for feat in FEATS:
            for scale in SCALE[i:]:
                self.up.update({f'{feat}_{scale}': UpSample(feat, scale, stride)})
            i += 1

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)

        self.selective_kernel = nn.ModuleList([SKFF(n_feat * stride ** i, height) for i in range(height)])

    def forward(self, x):
        inp = x.clone()
        # col 1 only
        blocks_out = []
        for j in range(self.height):
            if j == 0:
                inp = self.blocks[j][0](inp)
            else:
                inp = self.blocks[j][0](self.down[f'{inp.size(1)}_{2}'](inp))
            blocks_out.append(inp)

        # rest of grid
        for i in range(1, self.width):
            # Mesh
            # Replace condition(i%2!=0) with True(Mesh) or False(Plain)
            # if i%2!=0:
            if True:
                tmp = []
                for j in range(self.height):
                    TENSOR = []
                    nfeats = (2 ** j) * self.n_feat
                    for k in range(self.height):
                        TENSOR.append(self.select_up_down(blocks_out[k], j, k))

                    selective_kernel_fusion = self.selective_kernel[j](TENSOR)
                    tmp.append(selective_kernel_fusion)
            # Plain
            else:
                tmp = blocks_out
            # Forward through either mesh or plain
            for j in range(self.height):
                blocks_out[j] = self.blocks[j][i](tmp[j])

        # Sum after grid
        out = []
        for k in range(self.height):
            out.append(self.select_last_up(blocks_out[k], k))

        out = self.selective_kernel[0](out)

        out = self.conv_out(out)
        out = out + x

        return out

    def select_up_down(self, tensor, j, k):
        if j == k:
            return tensor
        else:
            diff = 2 ** np.abs(j - k)
            if j < k:
                return self.up[f'{tensor.size(1)}_{diff}'](tensor)
            else:
                return self.down[f'{tensor.size(1)}_{diff}'](tensor)

    def select_last_up(self, tensor, k):
        if k == 0:
            return tensor
        else:
            return self.last_up[f'{k}'](tensor)


# ---------- Recursive Residual Group (RRG) ----------
class RRG(nn.Module):
    def __init__(self, n_feat, n_MSRB=1, height=3, width=2, stride=2, bias=False):
        super(RRG, self).__init__()
        modules_body = [MSRB(n_feat, height, width, stride, bias) for _ in range(n_MSRB)]
        modules_body.append(default_conv(n_feat, n_feat, kernel_size=3))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


"""
20 groups by default. 
CA, RCAB by: 
Image Super-Resolution Using Very Deep Residual Channel Attention Networks

"""
# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=0.1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) * self.res_scale
        return res + x


# Residual Group (RG)
class RG(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=16, n_RCAB=15):
        # we use 10, it gives better results
        super().__init__()
        modules_body = [
            RCAB(
                n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_RCAB)]
        modules_body.append(default_conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x) + x
        return res


##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=False):
        super(SAM, self).__init__()
        self.conv1 = default_conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = default_conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = default_conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


#  Upsamler layer
class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, norm=False, act='relu', bias=False):

        m = []
        act = act_layer(act) if act else None
        norm = norm_layer(norm, n_feats) if norm else None
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
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


#  Upsamler layer
class UpsamplerSmall(nn.Sequential):
    def __init__(self, scale, n_feats, norm=False, act='relu', bias=False):

        m = []
        act = act_layer(act) if act else None
        norm = norm_layer(norm, n_feats) if norm else None
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.PixelShuffle(2))
                m.append(default_conv(n_feats // 4, n_feats, 3, bias=bias))
                if norm: m.append(norm)
                if act is not None: m.append(act)

        elif scale == 3:
            m.append(nn.PixelShuffle(3))
            m.append(default_conv(n_feats // 9, n_feats, 3, bias=bias))
            if norm: m.append(norm)
            if act is not None: m.append(act)
        else:
            raise NotImplementedError

        super(UpsamplerSmall, self).__init__(*m)


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


def raw_pack(input, scale=2):
    N, C, H, W = input.size()
    assert H % scale == 0, 'Please Check input and scale'
    assert W % scale == 0, 'Please Check input and scale'
    map_channels = scale ** 2
    channels = C * map_channels
    out_height = H // scale
    out_width = W // scale

    input_view = input.contiguous().view(
        N, C, out_height, scale, out_width, scale)

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(N, channels, out_height, out_width)


def raw_unpack(input):
    demo = nn.PixelShuffle(2)
    return demo(input)


#############################
#  counting number
#############################
def cal_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total
