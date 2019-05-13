import os
import argparse
import time
import shutil
import importlib
import numpy as np

import torch
from torch.utils.data import DataLoader

from datasets.load_dataset import LoadPixelShiftData
from model.common import print_model_parm_nums, demosaick_layer
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.LogTools.logger_tensorboard import Tf_Logger
from TorchTools.LossTools.loss import C_Loss, VGGLoss
from TorchTools.LossTools.metrics import PSNR, AverageMeter
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

    train_set = LoadPixelShiftData(args.train_list, args.patch_size, args.scale, args.denoise, args.max_noise, args.min_noise,
                           args.downsampler, args.get2label)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True)
    valid_set = LoadPixelShiftData(args.valid_list, args.patch_size, args.scale, args.denoise, args.max_noise, args.min_noise,
                           args.downsampler, args.get2label)
    valid_loader = DataLoader(dataset=valid_set, num_workers=8, batch_size=args.batch_size,
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
    # if args.vgg_lambda != 0:
    #     vggloss = VGGLoss(vgg_path=args.vgg_path, layers=args.vgg_layer, loss=args.vgg_loss)
    # else:
    #     vggloss = None

    if device.type == 'cuda':
        criterion = criterion.cuda()
        # if args.vgg_lambda != 0:
        #     vggloss.cuda()

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_adjust_freq, gamma=0.1)
    ###############################################################################################
    # load checkpoints
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
    sr_losses = AverageMeter()
    dm_losses = AverageMeter()
    # vgg_losses = AverageMeter()

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

            model.train()

            # train, data convert
            img = data['input'].to(device)
            gt = data['gt'].to(device)
            if args.get2label:
                raw_gt = data['raw_gt'].to(device)

            # zero parameters
            optimizer.zero_grad()

            # forward+backward+optim
            raw_output, output = model(img)

            # loss function
            if args.get2label:
                sr_loss = C_Loss(raw_output, raw_gt).cuda()
            else:
                sr_loss = 0.
                args.sr_lambda = 0.
            dm_loss = C_Loss(output, gt).cuda()

            # if args.vgg_lambda != 0:
            #     vgg_loss = vggloss(rgb_output, rgb_gt)
            # else:
            #     vgg_loss = 0
            # loss = args.dm_lambda * dm_loss + args.sr_lambda * sr_loss + args.vgg_lambda * vgg_loss
            loss = args.dm_lambda * dm_loss + args.sr_lambda * sr_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.update(loss.item(), gt.size(0))
            if args.get2label:
                sr_losses.update(sr_loss.item(), gt.size(0))
            else:
                sr_losses.update(sr_loss, gt.size(0))
            dm_losses.update(dm_loss.item(), gt.size(0))

            # if args.vgg_lambda != 0:
            #     vgg_losses.update(vgg_loss.item(), gt.size(0))



            if current_iter % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Iter:{3}\t'
                      'Loss {loss.val: .4f} ({loss.avg: .4f})\t'
                      'sr_loss {sr_loss.val: .4f} ({sr_loss.avg: .4f})\t'
                      'dm_loss {dm_loss.val: .4f} ({dm_loss.avg: .4f})\t'.format(
                       epoch, i, len(train_loader), current_iter, loss=losses,
                       sr_loss=sr_losses, dm_loss=dm_losses))


            # log
            info = {
                'loss': loss,
                'sr_loss': sr_loss,
                'dm_loss': dm_loss,
                'psnr': cur_psnr
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, current_iter)

            ######################################################################
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
            img = data['input'].to(device)
            gt = data['gt'].to(device)

            if args.get2label:
                raw_gt = data['raw_gt'].to(device)
            raw_output, output = model(img)

            # psnr
            mse = criterion(output, gt).item()
            psnr = PSNR(mse)
            psnrs.update(psnr, gt.size(0))
            print('Iter: [{0}][{1}/{2}]\t''TEST PSNR: ({psnrs.val: .4f}, {psnrs.avg: .4f})\t'.format(
                   iter, i, len(valid_loader), psnrs=psnrs))

    # show images
    img = img[0:1, 0:4]
    img = torch.clamp(img.cpu()[0, 0], 0, 1).detach().numpy()
    gt = torch.clamp(gt.cpu()[0], 0, 1).detach().numpy()
    gt = np.transpose(gt, [1, 2, 0]) 	# permute
    output = torch.clamp(output.cpu()[0], 0, 1).detach().numpy()
    output = np.transpose(output, [1, 2, 0])

    if args.get2label:
        raw_output = torch.clamp(raw_output.cpu()[0, 0], 0, 1).detach().numpy()
        raw_gt = torch.clamp(raw_gt.cpu()[0, 0], 0, 1).detach().numpy()
        vis = [img, raw_output, output, raw_gt, gt]
    else:
        vis = [img, output, gt]
    logger.image_summary(args.post, vis, iter)
    return psnrs.avg


############################################################################################
def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5


def save_checkpoint(state, is_best, args):
    filename = '{}/{}_checkpoint_{}k.path'.format(args.save_path, args.post, state['iter']/1000)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_model_best.path'.format(args.save_path, args.post))


if __name__=='__main__':
    main()
