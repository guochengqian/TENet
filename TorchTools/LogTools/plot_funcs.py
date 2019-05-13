import os
import pdb
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import random
import re
from torchvision import transforms
import torch
import cv2
import math
import numpy as np


# def parse_log(log_path, start=0, end=-1, interval=1):
#     """
#     parse log file
#     :param log_path:
#     :return: train / test loss list
#     """
#     log_file = open(log_path, 'r')
#     lines = log_file.readlines()
#     end = len(lines) if end == -1 else end
#     cnt = 0
#     train_loss = []
#     test_psnr = []
#     test_loss = []
#     for line in lines:
#         cnt += 1
#         if cnt > start and cnt < end and cnt % interval == 0:
#             line = line.strip().split(' ')
#             if line[0] == 'testset:':
#                 psnr = float(line[3])
#                 loss = float(line[5])
#                 test_psnr.append(psnr)
#                 test_loss.append(loss)
#             else:
#                 loss = float(line[2])
#                 train_loss.append(loss)
#     return train_loss, test_psnr, test_loss

test_prefix = ['test:', 'testset:']

def parse_log(log_path, start=0, end=-1, interval=1):
    """
    parse log file
    :param log_path:
    :return: train / test loss list
    """
    log_file = open(log_path, 'r')
    lines = log_file.readlines()
    end = len(lines) if end == -1 else end
    cnt = 0
    train_loss = []
    test_psnr = []
    test_loss = []
    for line in lines:
        cnt += 1
        if cnt > start and cnt < end and cnt % interval == 0:
            line = line.strip().split(' ')
            if line[0] in test_prefix:
                psnr = float(line[3])
                loss = float(line[5])
                test_psnr.append(psnr)
                test_loss.append(loss)
            else:
                loss = float(line[2])
                train_loss.append(loss)
    return train_loss, test_psnr, test_loss

def parse_RLSR_log(log_path, start=0, end=-1, interval=1):
    """
    parse log file: data iter loss: %d agent loss: %d %d %d %d
    :param log_path:
    :return: train / test loss list
    """
    log_file = open(log_path, 'r')
    lines = log_file.readlines()
    end = len(lines) if end == -1 else end
    cnt = 0
    train_init_loss = []
    train_loss = []
    test_psnr = []
    test_loss = []
    for line in lines:
        cnt += 1
        if cnt > start and cnt < end and cnt % interval == 0:
            line = line.strip().split(' ')
            if line[0] == 'testset:':
                psnr = float(line[3])
                loss = float(line[5])
                test_psnr.append(psnr)
                test_loss.append(loss)
            else:
                init_loss = float(line[3])
                loss = np.mean(np.array(line[-5:]).astype(np.float64))
                train_init_loss.append(init_loss)
                train_loss.append(loss)
    return train_init_loss, train_loss, test_psnr, test_loss

def plot_log(loss, phase, save=False, show=True, fsize=None):
    """
    plot loss fig from loss list
    :param loss:
    :param phase:
    :param save:
    :param show:
    :return:
    """
    x_loss = [i + 1 for i in range(len(loss))]
    linewidth = 0.75
    if phase == 'train':
        color = [0 / 255.0, 191 / 255.0, 255 / 255.0]
    else:
        color = [255 / 255.0, 106 / 255.0, 106 / 255.0]
    title = phase + '_loss'
    if fsize is not None:
        plt.figure(figsize=fsize)
    plt.plot(x_loss, loss, color=color, linewidth=linewidth)
    plt.title(title)
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    if save:
        plt.savefig(phase + '.png')
    if show:
        plt.show()





def show_img_from_tensor(tensor, save_dir='', show=False):
    """
    visualize tensor / Variable
    :param tensor: image
    :param show:
    :param save_dir: path to save image
    :return:
    """
    tensor2im = transforms.ToPILImage()
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, 0)
    im = tensor2im(tensor)
    if show:
        im.show()
    if save_dir != '':
        im.save(save_dir)


class Log(object):
    def __init__(self, log_path):
        self.log_path = log_path

        def _parse_log(log_path):
            log_file = open(log_path, 'r')
            lines = log_file.readlines()
            cnt = 0
            train_loss = []
            test_loss = []
            test_psnr = []
            for line in lines:
                cnt += 1
                if cnt > 10 and cnt % 1 == 0:
                    line = line.strip().split(' ')
                    if line[0] == 'testset:':
                        psnr = float(line[3])
                        loss = float(line[5])
                        test_psnr.append(psnr)
                        test_loss.append(loss)
                    else:
                        loss = float(line[2])
                        train_loss.append(loss)
            return train_loss, test_loss, test_psnr

        train_loss, test_loss, test_psnr = _parse_log(log_path)
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.test_psnr = test_psnr

    def plot_loss(self, label, save=False, show=True):

        def _plot(loss, label, save, show):
            x_loss = [i + 1 for i in range(len(loss))]
            linewidth = 0.75
            if label == 'train':
                color = [0 / 255.0, 191 / 255.0, 255 / 255.0]
            elif label == 'psnr':
                color = [255 / 255.0, 106 / 255.0, 106 / 255.0]
            else:
                color = [0 / 255.0, 106 / 255.0, 106 / 255.0]
            title = label + '_loss'
            plt.plot(x_loss, loss, color=color, linewidth=linewidth)
            plt.title(title)
            plt.xlabel('Iter')
            plt.ylabel('Loss')
            if save:
                plt.savefig(label + '.png')
            if show:
                plt.show()

        if label == 'train':
            _plot(self.train_loss, label, save, show)
        elif label == 'psnr':
            _plot(self.test_psnr, label, save, show)
        else:
            _plot(self.test_loss, label, save, show)

# def plot_log(log_path, phase='train', start=10, end=-1, interval=1, save=False, show=True):
#     """
#     log file fig
#     :param log_path:
#     :param phase:
#     :param save:
#     :param show:
#     :return:
#     """
#     train_loss, test_psnr, test_loss = parse_log(log_path, start, end)
#     if phase == 'train':
#         _plot(train_loss, phase, save, show)
#     elif phase == 'psnr':
#         _plot(test_psnr, phase, save, show)
#     else:
#         _plot(test_loss, phase, save, show)

