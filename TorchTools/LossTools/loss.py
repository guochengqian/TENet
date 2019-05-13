import torch
import pdb

import torch.nn
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
from torchvision import models
import torch.nn.init as init

import math
# import VGG

bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
# l1_loss = nn.L1Loss()

class Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, epsilon=1e-6):
        super(Charbonnier_loss, self).__init__()
        self.eps = epsilon ** 2

    def forward(self, X, Y):
        # batchsize = X.data.shape[0]
        diff = X - Y
        loss = torch.mean(torch.sqrt(diff ** 2 + self.eps)) 
        return loss


class L1_TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L1_TVLoss,self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_tv = torch.abs((x[:, :, 1:, :]-x[:, :, :-1, :])).sum()
        w_tv = torch.abs((x[:, :, :, 1:]-x[:, :, :, :-1])).sum()
        return h_tv + w_tv

class L1_TVLoss_Charbonnier(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L1_TVLoss_Charbonnier,self).__init__()
        self.e = 0.000001 ** 2

    def forward(self, x):
        batch_size = x.size()[0]
        h_tv = torch.abs((x[:, :, 1:, :]-x[:, :, :-1, :]))
        h_tv = torch.mean(torch.sqrt(h_tv ** 2 + self.e))
        w_tv = torch.abs((x[:, :, :, 1:]-x[:, :, :, :-1]))
        w_tv = torch.mean(torch.sqrt(w_tv ** 2 + self.e))
        return h_tv + w_tv

class TV_L1LOSS(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TV_L1LOSS,self).__init__()

    def forward(self, x, y):
        size = x.size()
        h_tv_diff = torch.abs((x[:, :, 1:, :] - x[:, :, :-1, :] - (y[:, :, 1:, :] - y[:, :, :-1, :]))).sum()
        w_tv_diff = torch.abs((x[:, :, :, 1:] - x[:, :, :, :-1] - (y[:, :, :, 1:] - y[:, :, :, :-1]))).sum()
        return (h_tv_diff + w_tv_diff) / size[0] / size[1] / size[2] / size[3]


class MSEloss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, epsilon=1e-3):
        super(MSEloss, self).__init__()
        self.eps = epsilon ** 2

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        sum_square_err = torch.sum(diff * diff)
        loss = sum_square_err / X.data.shape[0] / 2.
        # loss = torch.sum(error)
        return loss


class Perceptual_loss(nn.Module):
    def __init__(self, content_layer):
        super(Perceptual_loss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        model = nn.Sequential()
        i = 1
        j = 1
        temp = list(vgg)
        for layer in list(vgg):
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                model.add_module(name, layer)
                if name == content_layer:
                    break

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                model.add_module(name, layer)
                if name == content_layer:
                    break
                i += 1

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(j)
                model.add_module(name, layer)
                j += 1

        self.model = model
        self.criteria = nn.MSELoss()

    def forward(self, X, Y):
        X_content = self.model(X)
        Y_content = self.model(Y)
        loss = ((X_content - Y_content) ** 2).sum()
        loss /= (X_content.size()[0] * X_content.size()[1] * X_content.size()[2] * X_content.size()[3])
        return loss


def euclideanLoss(output, label, input_size):
    mse = mse_loss(output, label)
    mse = mse*((input_size)/2.)
    return mse

def euclideanLoss2(output, label):
    mse = MSEloss()(output, label)
    return mse

def L1NormLoss(output, label, input_size):
    l1norm = l1_loss(output, label)
    l1norm = l1norm*((input_size)/2.)
    return l1norm

def C_Loss(output, label):
    c_loss_func = Charbonnier_loss(epsilon=1e-3)
    return c_loss_func(output, label)

def TVLoss(output):
    l1_tvloss = L1_TVLoss()
    size = output.size()
    return l1_tvloss(output) / size[0] / size[1] / size[2] / size[3]

def TVLoss_Charbonnier(output):
    l1_tvloss = L1_TVLoss_Charbonnier()
    return l1_tvloss(output)

def TV_l1loss(output, label):
    tv_l1loss = TV_L1LOSS()
    return tv_l1loss(output, label)


def perception_loss_filter(output, label, var_bound, loss_network2, loss_network3):
    output_network2 = loss_network2(output)
    label_network2 = loss_network2(label)
    output_network3 = loss_network3(output_network2)
    label_network3 = loss_network3(label_network2)

    perception2 = mse_loss(output_network2, label_network2)
    perception3 = mse_loss(output_network3, label_network3)

    out_loss = perception2 + 2 * perception3
    return out_loss


def invert_preproc(imgs, white_level):
	# return sRGBforward(torch.transpose(imgs) / white_level)
	# print torch.min(white_level)
	a = sRGBforward(imgs / white_level)
	# print 'bala:', a.shape
	# print a
	return a

def sRGBforward(x):
	b = torch.Tensor([.0031308]).cuda()
	gamma = 1./2.4
	a = 1./(1./(b**gamma*(1.-gamma))-1.)
	k0 = (1+a)*gamma*b**(gamma-1.)
	gammafn = lambda x : (1+a)*torch.pow(torch.max(x, b), gamma)-a

	srgb = torch.where(x < b, k0 * x, gammafn(x))

	k1 = (1+a)*gamma
	srgb = torch.where(x > 1, k1 * x - k1 + 1, srgb)

	return srgb

def gradient(imgs):
	return torch.stack([0.5*(imgs[:,:,1:,:-1]-imgs[:,:,:-1,:-1]), 
		0.5*(imgs[:,:,:-1,1:]-imgs[:,:,:-1,:-1])], dim=-1)

def gradient_loss(img, truth):
	gi = gradient(img)
	gt = gradient(truth)

	sh = gi.shape
	# print 'sh', sh
	length = 1
	for i in range(len(sh)):
		length *= sh[i]

	return torch.sum(torch.abs(gi - gt)) / length

def basic_img_loss(img, truth):
	# pdb.set_trace()
	sh = img.shape
	# print 'sh', sh
	length = 1
	for i in range(len(sh)):
		length *= sh[i]
	l2_pixel = torch.sum((img - truth)*(img-truth)) / length
	l1_grad = gradient_loss(img, truth)
	# print 'l2_pixel', l2_pixel
	return l2_pixel + l1_grad

# class VGGLoss(nn.Module):
#     def __init__(self):
#         super(VGGLoss, self).__init__()        
#         self.vgg = Vgg19()
#         self.criterion = nn.L1Loss()
#         self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

#     def forward(self, x, y):              
#         x_vgg, y_vgg = self.vgg(x), self.vgg(y)
#         loss = 0
#         for i in range(len(x_vgg)):
#             loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
#         return loss

# from torchvision import models
# from torchvision.models.vgg import model_urls

# class Vgg19(torch.nn.Module):
#     def __init__(self, requires_grad=False):
#         super(Vgg19, self).__init__()
#         # model_urls['vgg19'] = model_urls['vgg19'].replace('https://', 'http://')
#         model = models.vgg19(pretrained=False)
#         model.load_state_dict(torch.load('vgg19.pth'))
#         vgg_pretrained_features = model.features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(2):
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(2, 7):
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(7, 12):
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 21):
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(21, 30):
#             self.slice5.add_module(str(x), vgg_pretrained_features[x])
#         if not requires_grad:
#             for param in self.parameters():
#                 param.requires_grad = False

#     def forward(self, X):
#         h_relu1 = self.slice1(X)
#         h_relu2 = self.slice2(h_relu1)        
#         h_relu3 = self.slice3(h_relu2)        
#         h_relu4 = self.slice4(h_relu3)        
#         h_relu5 = self.slice5(h_relu4)                
#         out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
#         return out

def gaussian2d(u, v, sigma):
    pi = 3.1416
    intensity = 1 / (2.0 * pi * sigma * sigma) * math.exp(- 1 / 2.0 * ((u ** 2) + (v ** 2)) / (sigma ** 2))
    return intensity

def gaussianKernal(r, sigma):
    kernal = np.zeros([r, r])
    center = (r - 1) / 2.0
    for i in range(r):
        for j in range(r):
            kernal[i, j] = gaussian2d(i - center, j - center, sigma)
    kernal /= np.sum(np.sum(kernal))
    return kernal

def weights_init_Gaussian_blur(sigma=1.0):
    def sub_func(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            weight_shape = m.weight.data.size()
            gaussian_blur = gaussianKernal(weight_shape[2], sigma)
            for i in range(weight_shape[0]):
                m.weight.data[i, 0, :, :] = torch.from_numpy(gaussian_blur)
            if not m.bias is None:
                m.bias.data.zero_()
    return sub_func

def weights_init_He_normal(m):
    classname = m.__class__.__name__
#     print classname
    if classname.find('Transpose') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Conv') != -1:
        # std = np.sqrt(2. / (m.kernel_size[0] * m.kernel_size[1] * m.out_channels))
        # m.weight.data.normal_(0.0, std)
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        if not m.bias is None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        if not m.bias is None:
            m.bias.data.zero_()

def l1_loss(input, output):
    return torch.mean(torch.abs(input-output))


class VGGLoss(nn.Module):
    """
    VGG(
    (features): Sequential(
    (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)
    (18): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (19): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (24): ReLU(inplace)
    (25): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (26): ReLU(inplace)
    (27): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

    (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): ReLU(inplace)
    (32): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (33): ReLU(inplace)
    (34): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): ReLU(inplace)
    (36): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
    )

    """
    def __init__(self, vgg_path, layers='45', input='RGB', loss='l1'):
        super(VGGLoss, self).__init__()
        self.input = input
        vgg = models.vgg19(pretrained=False)
        if vgg_path is not '':
            vgg.load_state_dict(torch.load(vgg_path))
            # vgg.load_state_dict(torch.load('../../../'+vgg_path))
        self.layers = [int(l) for l in layers]
        layers_dict = [0, 4, 9, 18, 27, 36]
        self.vgg = []
        if loss == 'l1':
            self.loss_func = l1_loss
            # pytorch 0.4 l1_loss malfunction
        elif loss == 'l2':
            self.loss_func = nn.functional.mse_loss
        else:
            raise Exception('Do not support this loss.')

        i = 0
        for j in self.layers:
            self.vgg.append(nn.Sequential(*list(vgg.features.children())[layers_dict[i]:layers_dict[j]]))
            i = j

    def cuda(self, device=None):
        for Seq in self.vgg:
            Seq.cuda()

    def forward(self, input, target):
        if self.input == 'RGB':
            input_R, input_G, input_B = torch.split(input, 1, dim=1)
            target_R, target_G, target_B = torch.split(target, 1, dim=1)
            input_BGR = torch.cat([input_B, input_G, input_R], dim=1)
            target_BGR = torch.cat([target_B, target_G, target_R], dim=1)
        else:
            input_BGR = input
            target_BGR = target

        # pdb.set_trace()
        input_list = [input_BGR]
        target_list = [target_BGR]

        for Sequential in self.vgg:
            input_list.append(Sequential(input_list[-1]))
            target_list.append(Sequential(target_list[-1]))

        # pdb.set_trace()        
        loss = []
        for i in range(len(self.layers)):
            loss.append(self.loss_func(input_list[i + 1], target_list[i + 1]))
        # pdb.set_trace()  
        return sum(loss)
