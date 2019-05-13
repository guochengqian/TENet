import os
import argparse

import shutil
import importlib
import torch
from torch.utils.data import DataLoader

from datasets.load_dataset import LoadRawDeno, LoadRgbDeno, LoadDemo, LoadRawSR, LoadRgbSR, LoadPixelShiftData, LoadSimData
from model.common import print_model_parm_nums
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.LogTools.logger_tensorboard import Tf_Logger
from TorchTools.LossTools.loss import C_Loss
from TorchTools.LossTools.metrics import PSNR, AverageMeter
import numpy as np
import pdb

def main():
    ###############################################################################################
    # args parse
    parser = argparse.ArgumentParser(description='PyTorch implementation of DDSR')
    parsers = BaseArgs()
    args = parsers.initialize(parser)
    parsers.print_args()

    ###############################################################################################
    print('===> Creating dataloader...')
    # in_channels, gt_channels
    if args.model == 'denorgb':
        train_set = LoadRgbDeno(args.train_list, args.patch_size, args.max_noise, args.min_noise)
        valid_set = LoadRgbDeno(args.valid_list, args.patch_size, args.max_noise, args.min_noise)
    elif args.model == 'denoraw':
        train_set = LoadRawDeno(args.train_list, args.patch_size, args.max_noise, args.min_noise)
        valid_set = LoadRawDeno(args.valid_list, args.patch_size, args.max_noise, args.min_noise)
    elif args.model == 'demo':
        train_set = LoadDemo(args.train_list, args.patch_size, args.denoise, args.max_noise)
        valid_set = LoadDemo(args.valid_list, args.patch_size, args.denoise, args.max_noise)
    elif args.model == 'srraw':
        train_set = LoadRawSR(args.train_list, args.patch_size, args.scale, args.denoise, args.max_noise)
        valid_set = LoadRawSR(args.valid_list, args.patch_size, args.scale, args.denoise, args.max_noise)
    elif args.model == 'srrgb':
        train_set = LoadRgbSR(args.train_list, args.patch_size, args.scale, args.denoise, args.max_noise)
        valid_set = LoadRgbSR(args.valid_list, args.patch_size, args.scale, args.denoise, args.max_noise)
    elif args.model == 'tri1':
        train_set = LoadSimData(args.train_list, args.patch_size, args.scale, args.denoise, args.max_noise)
        valid_set = LoadSimData(args.valid_list, args.patch_size, args.scale, args.denoise, args.max_noise)
    else:
        raise ValueError('not supported model')

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_set, num_workers=4, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True)

    ###############################################################################################
    print('===> Loading the network ...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args).to(device)
    print(model)
    print_model_parm_nums(model)

    ###############################################################################################
    # optimize
    criterion = torch.nn.MSELoss()

    if device.type == 'cuda':
        criterion = criterion.cuda()

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_adjust_freq, gamma=0.1)
    ###############################################################################################
    # load pre-trained
    best_psnr = 0
    cur_psnr = 0
    if args.pretrained_model:
        if os.path.isfile(args.pretrained_model):
            print("=====> loading checkpoint '{}'".format(args.pretrained_model))
            checkpoint = torch.load(args.pretrained_model)
            best_psnr = checkpoint['best_psnr']
            model.load_state_dict(checkpoint['state_dict'])
            print("The pretrained_model is at checkpoint {}, and it's best loss is {}."
                  .format(checkpoint['iter'], best_psnr))
        else:
            print("=====> no checkpoint found at '{}'".format(args.pretrained_model))

    ###############################################################################################
    # train + valid
    logger = Tf_Logger(args.logdir)
    losses = AverageMeter()
    current_iter = args.start_iters

    print('---------- Start training -------------')
    for epoch in range(args.total_epochs):
        ###########################################
        # train
        for i, data in enumerate(train_loader):
            current_iter += 1
            if current_iter > args.total_iters:
                break

            # valid
            if current_iter % args.valid_freq == 0:
                cur_psnr = valid(valid_loader, model, criterion, current_iter, args, device, logger)

            # train, data convert
            # train, data convert
            model.train()
            img = data['input']
            gt = data['gt']
            img, gt= img.to(device), gt.to(device)

            # zero parameters
            optimizer.zero_grad()

            # forward+backward+optim
            output = model(img)

            # loss function
            loss = C_Loss(output, gt).cuda()
            # optim
            loss.backward()
            optimizer.step()
            scheduler.step()

            # info upateTestArgs
            losses.update(loss.item(), gt.size(0))

            if current_iter % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Iter:{3}\t'
                      'Loss {loss.val: .4f} ({loss.avg: .4f})\t'.format(
                       epoch, i, len(train_loader), current_iter, loss=losses))
            #######################################################################
            # log
            info = {'loss': losses.avg, 'psnr': cur_psnr}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, current_iter)

            # save checkpoints
            if current_iter % args.save_freq == 0:
                is_best = (cur_psnr > best_psnr)
                best_psnr = max(cur_psnr, best_psnr)
                model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
                save_checkpoint({
                    'iter': current_iter,
                    'state_dict': model_cpu,
                    'best_psnr': best_psnr
                }, is_best, args=args)

    print('Saving the final model.')


############################################################################################
#
#   functions
#
############################################################################################
def valid(valid_loader, model, criterion, iter, args, device, logger):
    psnrs = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            # valid
            img = data['input']
            gt = data['gt']
            img, gt = img.to(device), gt.to(device)

            # pdb.set_trace()
            output = model(img.detach())

            # psnr
            mse = criterion(output, gt).item()
            psnr = PSNR(mse)
            psnrs.update(psnr, gt.size(0))
            print('Iter: [{0}][{1}/{2}]\t''TEST PSNR: ({psnrs.val: .4f}, {psnrs.avg: .4f})\t'.format(
                   iter, i, len(valid_loader), psnrs=psnrs))

    # show images
    if img.shape[1] == 3:
        img = torch.clamp(img.cpu()[0], 0, 1).detach().numpy()
        img = np.transpose(img, [1, 2, 0])
    else:
        img = torch.clamp(img.cpu()[0, 0], 0, 1).detach().numpy()

    if gt.shape[1] == 3:
        gt = torch.clamp(gt.cpu()[0], 0, 1).detach().numpy()
        output = torch.clamp(output.cpu()[0], 0, 1).detach().numpy()
        gt = np.transpose(gt, [1, 2, 0])
        output = np.transpose(output, [1, 2, 0])
    else:
        gt = torch.clamp(gt.cpu()[0, 0], 0, 1).detach().numpy()
        output = torch.clamp(output.cpu()[0, 0], 0, 1).detach().numpy()

    vis = [img, output, gt]
    del gt, img, output
    logger.image_summary(args.post, vis, iter)
    return psnrs.avg


############################################################################################
def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.5 ** (epoch // args.step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5


def save_checkpoint(state, is_best, args):
    filename = '{}/{}_checkpoint_{}k.path'.format(args.save_path, args.post, state['iter']/1000)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_model_best.path'.format(args.save_path, args.post))


if __name__ == '__main__':
    main()


