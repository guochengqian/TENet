import argparse
import shutil
import importlib
import logging
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import process
from datasets.load_dataset import LoadSimDataUnproc, LoadPixelShiftData
from model.common import cal_model_parm_nums
from TorchTools.ArgsTools.base_args import BaseArgs
from TorchTools.LossTools.metrics import PSNR, AverageMeter
from TorchTools.model_util import load_pretrained_models, load_pretrained_optimizer
from torch.utils.tensorboard import SummaryWriter


def main():
    logging.info('===> Creating dataloader...')
    if 'pixelshift' in args.dataset:
        train_set = LoadPixelShiftData(args.train_list, 'train',
                                       args.patch_size,
                                       args.downsampler, args.scale,
                                       args.in_type, args.mid_type, args.out_type)
        val_set = LoadPixelShiftData(args.val_list, 'val',
                                     args.patch_size,
                                     args.downsampler, args.scale,
                                     args.in_type, args.mid_type, args.out_type)
    else:
        train_set = LoadSimDataUnproc(args.train_list, 'train',
                                      args.patch_size,
                                      args.downsampler, args.scale,
                                      args.in_type, args.mid_type, args.out_type)
        val_set = LoadSimDataUnproc(args.val_list, 'val',
                                    args.patch_size,
                                    args.downsampler, args.scale,
                                    args.in_type, args.mid_type, args.out_type)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True)

    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=args.batch_size,
                            shuffle=False, pin_memory=True)

    # =================
    logging.info('===> Loading the network ...')
    module = importlib.import_module("model.{}".format(args.model))
    model = module.NET(args)
    # =================
    # load checkpoints
    if args.pretrain_other:
        model.load_state_dict_from_other_pipeline(args.pretrain_other)
    model, best_psnr, start_epoch = load_pretrained_models(model, args.pretrain)

    if args.n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(args.device)

    logging.info(model)
    model_size = cal_model_parm_nums(model)
    logging.info('Number of params: %.4f MB' % (model_size / (8 * 1e6)))

    # =================
    logging.info('===> Loading the Optimizers ...')
    criterion = torch.nn.L1Loss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    elif args.lr_scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, args.lr * 1e-3)
    else:
        raise NotImplementedError(f'{args.lr_scheduler} is not supported. support step or cos lr scheduler. ')
    optimizer, scheduler = load_pretrained_optimizer(args.pretrain, optimizer, scheduler, args.device)

    # =================
    # train + val
    logging.info('---------- Start training -------------\n')
    iters = len(train_loader)
    last_loss = np.inf

    for epoch in range(start_epoch, args.max_epochs):
        # train
        losses = AverageMeter()
        mid_losses = AverageMeter()
        main_losses = AverageMeter()
        model.train()
        for i, data in enumerate(train_loader):
            # train, data convert
            if 'noisy' in args.in_type:
                img = torch.cat((data[args.in_type], data['variance']), dim=1).to(args.device)
            else:
                img = data[args.in_type].to(args.device)

            gt = data[args.out_type].to(args.device)
            batch_size = gt.size(0)

            if args.mid_type is not None:
                mid_output, output = model(img)
            else:
                output = model(img)

            # forward+backward+optimization
            main_loss = criterion(output, gt)
            if main_loss > 4 * last_loss:
                continue

            if args.mid_type is not None:
                mid_gt = data[args.mid_type].to(args.device)
                mid_loss = criterion(mid_output, mid_gt)
                loss = main_loss + args.mid_lambda * mid_loss
            else:
                loss = main_loss

            # zero parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), batch_size)
            main_losses.update(main_loss.item(), batch_size)
            if args.mid_type is not None:
                mid_losses.update(mid_loss.item(), batch_size)

            if i % args.print_freq == 0:
                if args.mid_type is not None:
                    logging.info('Epoch: [{}]/[{}] Iter:[{}]/[{}]\t'
                                 'Loss {loss.val:.4f} \t'
                                 'main_loss {main_loss.val:.4f}\t'
                                 'mid_loss {mid_loss.val:.4f}'.format(
                                  epoch, args.max_epochs, i, iters,
                                  loss=losses, main_loss=main_losses, mid_loss=mid_losses))
                else:
                    logging.info('Epoch: [{}]/[{}] Iter:[{}]/[{}]\t'
                                 'Loss {loss.val:.4f}'.format(
                                  epoch, args.max_epochs, i, iters, loss=losses))
            last_loss = main_loss

        scheduler.step()
        # loss_last = losses.avg
        writer.add_scalar('train_loss', losses.avg, epoch)
        writer.add_scalar('main_loss', main_losses.avg, epoch)
        writer.add_scalar('mid_loss', mid_losses.avg, epoch)
        writer.add_scalar('lr', scheduler.get_last_lr()[-1], epoch)

        # ================
        # val
        if epoch % args.eval_freq == 0 or epoch == args.max_epochs - 1:
            args.epoch = epoch
            cur_psnr = val(val_loader, model, args.vis_eval)
            is_best = (cur_psnr > best_psnr)
            best_psnr = max(cur_psnr, best_psnr)
            model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model_cpu,
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best, args=args)
            writer.add_scalar('eval_psnr', cur_psnr, epoch)

    logging.info('Saving the final model.')

    # wandb
    if args.use_wandb:
        args.Wandb.add_file(f'{args.ckpt_dir}/{args.jobname}_checkpoint_best.pth')
        args.Wandb.add_file(f'{args.ckpt_dir}/{args.jobname}_checkpoint_latest.pth')


def val(val_loader, model, vis_eval=False):
    psnrs = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if 'noisy' in args.in_type:
                img = torch.cat((data[args.in_type], data['variance']), dim=1).to(args.device)
            else:
                img = data[args.in_type].to(args.device)

            gt = data[args.out_type].to(args.device)

            if args.mid_type is not None:
                mid_output, output = model(img)
            else:
                output = model(img)

            # psnr
            mse = (output.clamp(0, 1) - gt).pow(2).mean()
            psnr = PSNR(mse)
            psnrs.update(psnr, gt.size(0))

            if i == 0 and vis_eval and (args.epoch % args.img_freq == 0):
                batch_size = img.shape[0]
                n_img = min(5, batch_size)
                n_stride = batch_size // n_img  # show 5 imgs only

                if 'rgb' in args.in_type and 'linrgb' not in args.out_type:
                    rgb_out = output[::n_stride]
                    rgb_gt = gt[::n_stride]
                else:
                    red_g = data['metadata']['red_gain'][::n_stride].to(args.device)
                    blue_g = data['metadata']['blue_gain'][::n_stride].to(args.device)
                    ccm = data['metadata']['ccm'][::n_stride].to(args.device)

                    if 'raw' in args.out_type:
                        rgb_out = process.raw2srgb(output[::n_stride], red_g, blue_g, ccm)
                        rgb_gt = process.raw2srgb(gt[::n_stride], red_g, blue_g, ccm)
                    elif 'linrgb' in args.out_type:
                        rgb_out = process.rgb2srgb(output[::n_stride], red_g, blue_g, ccm)
                        rgb_gt = process.rgb2srgb(gt[::n_stride], red_g, blue_g, ccm)

                B, C, H, W = rgb_out.shape
                writer.add_images('eval_result',
                                  torch.stack((rgb_gt, rgb_out), dim=1).contiguous().view(-1, C, H, W), args.epoch)
    logging.info('\nEpoch: [{}]/[{}] \t''TEST PSNR: {psnrs.avg: .4f})\n'.
                 format(args.epoch, args.max_epochs, psnrs=psnrs))

    return psnrs.avg


def save_checkpoint(state, is_best, args):
    filename = '{}/{}_checkpoint_latest.pth'.format(args.ckpt_dir, args.jobname)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_checkpoint_best.pth'.format(args.ckpt_dir, args.jobname))
    if args.epoch % args.epoch_freq == 0:
        shutil.copyfile(filename, '{}/{}_checkpoint_epoch{}.pth'.format(args.ckpt_dir, args.jobname, args.epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of ISP-Net')
    args = BaseArgs(parser).args
    writer = SummaryWriter(log_dir=args.exp_dir)
    main()

# below is the code for debug:
# from datasets import process
# def vis_rgb(x):
#     import matplotlib.pyplot as plt
#     plt.imshow(x)
#     plt.show()
# vis_rgb(gt[0].permute(1,2,0).cpu())

# gt_srgb = process.rgb2srgb(gt, data['red_gain'].to(args.device), data['blue_gain'].to(args.device), data['cam2rgb'].to(args.device))
# vis_rgb(gt_srgb[0].permute(1,2,0).cpu())

# noisy_rgb = process.raw2srgb(data['input'].to(args.device), data['red_gain'].to(args.device), data['blue_gain'].to(args.device), data['cam2rgb'].to(args.device))
# vis_rgb(noisy_rgb[0].permute(1,2,0).cpu())

# clean_rgb = process.raw2srgb(data['mid_gt'].to(args.device), data['red_gain'].to(args.device), data['blue_gain'].to(args.device), data['cam2rgb'].to(args.device))
# vis_rgb(clean_rgb[0].permute(1,2,0).cpu())


# Debug the RawDeno:
# def vis_rgb(x):
#     import matplotlib.pyplot as plt
#     plt.imshow(x)
#     plt.show()
# red_g = data['metadata']['red_gain'][0:1].to(args.device)
# blue_g = data['metadata']['blue_gain'][0:1].to(args.device)
# ccm = data['metadata']['cam2rgb'][0:1].to(args.device)
# rgb_in = process.raw2srgb(img[0:1, :4], red_g, blue_g, ccm)
# rgb_gt = process.raw2srgb(gt[0:1], red_g, blue_g, ccm)
# vis_rgb(rgb_in[0].permute(1,2,0).detach().cpu().numpy())
# vis_rgb(rgb_gt[0].permute(1,2,0).detach().cpu().numpy())
# if debug:
# red_g = torch.tensor(2.7385).to('cuda')
# blue_g = torch.tensor(1.3687).to('cuda')
# ccm = torch.tensor([[[1.7365, -0.5612, -0.1753], [-0.1531, 1.5584, -0.4053], [0.0199, -0.4041, 1.3842]]],
#                    device='cuda')
# rgb_out = process.rgb2srgb(output, red_g, blue_g, ccm)

# # debug:
# red_g, blue_g, ccm = data['metadata']['red_gain'].to(args.device), \
#                      data['metadata']['blue_gain'].to(args.device), \
#                      data['metadata']['ccm'].to(args.device)
# rgb_gt = process.rgb2srgb(gt, red_g, blue_g, ccm)
# out_gt = process.rgb2srgb(output, red_g, blue_g, ccm)
