import os
import os.path as osp
import sys
import time
from typing import List
import logging
import pathlib
import uuid
import shutil
import torch
import random
import numpy as np
from easydict import EasyDict as edict
from TorchTools.LogTools import Wandb


class BaseArgs:
    def __init__(self, parser):
        parser.add_argument('--phase', type=str, default='train',
                            help='phase. Default: train')
        parser.add_argument('--seed', type=int, default=0)

        # datasets args
        parser.add_argument('--dataset', type=str, default='pixelshift',
                            help='name of dataset to use')
        parser.add_argument('--test_dataset', type=str, default='urban',
                    help='name of dataset to use in testing')
        parser.add_argument('--patch_size', default=128, type=int,
                            help='width and height for a patch (default: 128); '
                                 'if performing joint DM and SR, then use 128.')
        parser.add_argument('--val_patch_size', default=128, type=int,
                            help='width and height for a patch (default: 128); '
                                 'if performing joint DM and SR, then use 128.')
        parser.add_argument('--in_type', type=str, default='noisy_lr_raw',
                            help='the input image type: noisy_lr_raw, lr_raw, noisy_raw, raw, '
                                 'noisy_lr_linrgb, lr_linrgb, noisy_linrgb, linrgb, '
                                 'noisy_lr_rgb, lr_rgb, noisy_rgb, rgb'
                            )
        parser.add_argument('--mid_type', type=str, default='None',
                            help='the mid output image type: noisy_lr_raw, lr_raw, noisy_raw, raw, '
                                 'noisy_lr_linrgb, lr_linrgb, noisy_linrgb, linrgb, '
                                 'noisy_lr_rgb, lr_rgb, noisy_rgb, rgb, None'
                            )
        parser.add_argument('--out_type', type=str, default='linrgb',
                            help='the output image type: noisy_lr_raw, lr_raw, noisy_raw, raw, '
                                 'noisy_lr_linrgb, lr_linrgb, noisy_linrgb, linrgb, '
                                 'noisy_lr_rgb, lr_rgb, noisy_rgb, rgb'
                            )
        parser.add_argument('--output_mid', action='store_true',
                            help='output the middle stage result')
        parser.add_argument('--noise_model', default='gp', type=str,
                            help='noise model, using gaussian-possion by default')
        parser.add_argument('--sigma', default=10, type=int,
                            help='sigam of the noise when noise model is gaussian')
        # train args
        parser.add_argument('--batch_per_gpu', default=32, type=int,
                            help='batch size per GPU (default:32)')
        parser.add_argument('--n_gpus', default=1, type=int,
                            help='number of GPUs (default:1)')
        parser.add_argument('--max_epochs', default=1000, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=5e-4, type=float,
                            help='initial learning rate')
        parser.add_argument('--scheduler', default='cos', type=str,
                            help='learning rate decay scheduler (step | cos)')
        parser.add_argument('--gamma', default=0.99, type=float, help='gamma for lr decay')
        # logger parse
        parser.add_argument('--root_dir', type=str, default='log',
                            help='path for saving experiment files')
        parser.add_argument('--vis_eval', default=False, action='store_true',
                            help='generate evaluation result (images) and upload to tensorboard')
        parser.add_argument('--img_freq', default=100, type=int,
                            help='show images every xxx epochs(default: 100)')
        parser.add_argument('--print_freq', default=100, type=int,
                            help='show loss information every xxx iterations(default: 100)')
        parser.add_argument('--eval_freq', default=10, type=int,
                            help='perform evaluation every xxx epochs(default: 20)')
        parser.add_argument('--save_freq', default=-1, type=int,
                            help='save milestone epoch every xxx epochs'
                                 'negative means only save latest and best (default: -1)')
        parser.add_argument('--use_wandb', action='store_true',
                            help='set this to use wandb or online logging')
        # model args
        parser.add_argument('--model', default='tenet', type=str,
                            help='model type (default: tenetv2)')
        parser.add_argument('--norm', default=None, type=str,
                            help='normalization_type(default: do not use BN or IN)')
        parser.add_argument('--block', default='rrdb', type=str,
                            help='dm_block(default: res). res/dudb/rrdb')
        parser.add_argument('--act', default='relu', type=str,
                            help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--no_bias', action='store_false', dest='bias',
                            help='do not use bias of layer')
        parser.add_argument('--channels', default=64, type=int,
                            help='channels')
        parser.add_argument('--n_blocks', default=12, type=int, nargs='+',
                            # parser.add_argument('--n_blocks', default=12, type=int,
                            help='number of basic blocks')
        # for super-resolution
        parser.add_argument('--scale', default=2, type=int,
                            help='Scale of Super-resolution. Default: 2')
        parser.add_argument('--downsampler', default='bic', type=str,
                            help='downsampler of Super-resolution. Bicubic or average downsampling.  bic / avg')
        # loss args
        parser.add_argument('--mid_lambda', type=float, default=1.0,
                            help='lamda for the middle stage supervision')
        parser.add_argument('--grad_norm_clip', default=1.,
                            type=float, help='clip gradient')
        parser.add_argument('--skip_threshold', type=float, default=5,
                            help='skip the batch is the loss is too large')
        parser.add_argument('--loss_on_srgb', action='store_true',
                            help='calculate the loss function values on sRGB')
        # test args
        parser.add_argument('--pred_dir', type=str, default=None,
                            help='path to prediction')
        parser.add_argument('--pred_pattern', type=str, default='',
                            help='the pattern of prediction file')
        parser.add_argument('--pretrain', default=None, type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--pretrain_other', default='', type=str,   # TODO: whats this ?
                            help='path to pretrained of other pipeline')
        
        # for testing        
        parser.add_argument('--intermediate', type=bool, default=False,
                            help='ISP intermediate state')
        parser.add_argument('--pre_in_type', type=str, default='noisy_lr_raw',
                            help='the input image type: noisy_lr_raw, lr_raw, noisy_raw, raw, '
                                 'noisy_lr_linrgb, lr_linrgb, noisy_linrgb, linrgb, '
                                 'noisy_lr_rgb, lr_rgb, noisy_rgb, rgb'
                            )
        parser.add_argument('--pre_out_type', type=str, default='raw',
                            help='the output image type: noisy_lr_raw, lr_raw, noisy_raw, raw, '
                                 'noisy_lr_linrgb, lr_linrgb, noisy_linrgb, linrgb, '
                                 'noisy_lr_rgb, lr_rgb, noisy_rgb, rgb'
                            )
        parser.add_argument('--pre_model', default='tenet', type=str,
                            help='path to pretrained model (default: tenet)')
        parser.add_argument('--save_dir', default=None, type=str)        
        args = parser.parse_args()

        # data related
        args.train_list = f'datasets/train_{args.dataset}.txt'
        args.val_list = f'datasets/val_{args.dataset}.txt'
        args.benchmark_path = f'data/benchmark/{args.test_dataset}'
        args.gt_dir = f'data/benchmark/{args.test_dataset}/gt'
        if args.save_dir is None:
            args.save_dir = f'results/{args.dataset}/{args.test_dataset}/{args.noise_model}/{args.model}-{args.in_type}-{args.mid_type}-{args.out_type}-SR{args.scale}'
        if args.save_dir is not None:
            os.makedirs(args.save_dir, exist_ok=True)
        
        args.in_channels = 3 if 'raw' not in args.in_type else 4
        args.gt_channels = 3 if 'raw' not in args.out_type else 4
        args.noise_channels = args.in_channels if 'p' in args.noise_model else 1
        args.batch_size = args.batch_per_gpu * args.n_gpus
        args.mid_lambda = [args.mid_lambda]*len(args.mid_type) if not isinstance(
            args.mid_lambda, List) else args.mid_lambda
        if isinstance(args.n_blocks, List) and len(args.n_blocks) == 1:
            args.n_blocks = args.n_blocks[-1]
        args.pre_pipename = '-'.join([args.pre_in_type, args.mid_type, args.pre_out_type]) 
        args.pipename = '-'.join([args.in_type, args.mid_type, args.out_type]) 
        args.expprefix = '-'.join([args.pipename, 
                                    args.model, args.dataset, args.block,
                                    'n' + str(args.n_blocks)])
        args.jobname = '-'.join([args.expprefix,
                                 'SR' + str(args.scale),
                                 f'noise_{args.noise_model}', 
                                 f'sigma{args.sigma}', 
                                 'C' + str(args.channels),
                                 'B' + str(args.batch_size),
                                 'Patch' + str(args.patch_size),
                                 'Epoch' + str(args.max_epochs)])
        if args.loss_on_srgb:
            args.jobname += '-loss_on_srgb'
        if args.mid_type.lower() == 'none':
            args.mid_type = None
        else:
            args.mid_type = str(args.mid_type).lower().split(',')
        self.args = args
        self.args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # ===> generate log dir
        if self.args.phase == 'train':
            # generate exp_dir when pretrained model does not exist, otherwise continue training using the pretrained
            if not self.args.pretrain:
                self._generate_exp_directory()
            else:
                self.args.exp_name = os.path.basename(
                    os.path.dirname(os.path.dirname(self.args.pretrain)))
                self.args.exp_dir = os.path.dirname(
                    os.path.dirname(self.args.pretrain))
                self.args.ckpt_dir = os.path.join(
                    self.args.exp_dir, "checkpoint")

            # set some value to Training mode
            self.args.output_mid = True if self.args.mid_type is not None else False

        elif not self.args.phase == 'debug':
            self.args.exp_dir = os.path.dirname(args.pretrain)
            # self.args.res_dir = os.path.join(os.path.dirname(self.args.exp_dir), 'result')
            # pathlib.Path(self.args.res_dir).mkdir(parents=True, exist_ok=True)

        if not self.args.phase == 'debug':
            self._configure_logger()
            self._configure_wandb()
        self.set_seed(self.args.seed)

    def _generate_exp_directory(self):
        """
        Helper function to create checkpoint folder. We save
        model checkpoints using the provided model directory
        but we add a sub-folder for each separate experiment:
        """
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.args.exp_name = '-'.join([self.args.jobname,
                                      timestamp, str(uuid.uuid4())])
        self.args.exp_dir = osp.join(self.args.root_dir, self.args.exp_name)
        self.args.ckpt_dir = osp.join(self.args.exp_dir, "checkpoint")
        pathlib.Path(self.args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    def _configure_logger(self):
        """
        Configure logger on given level. Logging will occur on standard
        output and in a log file saved in model_dir.
        """
        self.args.loglevel = "info"
        numeric_level = getattr(logging, self.args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(
                'Invalid log level: {}'.format(self.args.loglevel))

        log_format = logging.Formatter('%(asctime)s %(message)s')
        logger = logging.getLogger()
        logger.setLevel(numeric_level)

        file_handler = logging.FileHandler(osp.join(self.args.exp_dir,
                                                    '{}.log'.format(osp.basename(self.args.exp_dir))))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        file_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.root = logger
        logging.info(
            "save log, checkpoint and code to: {}".format(self.args.exp_dir))

    def _configure_wandb(self):
        if self.args.use_wandb:
            self.args.wandb = edict()
            self.args.wandb.entitiy = 'guocheng-qian'
            self.args.wandb.tags = self.args.jobname.split('-')
            self.args.wandb.name = self.args.exp_name
            self.args.Wandb = Wandb
            self.args.Wandb.launch(self.args, self.args.use_wandb)

    def _print_args(self):
        logging.info("==========       args      =============")
        for arg, content in self.args.__dict__.items():
            logging.info("{}: {}".format(arg, content))
        logging.info("==========     args END    =============")
        logging.info("\n")
        logging.info('===> Phase is {}.'.format(self.args.phase))

    @staticmethod
    def set_seed(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # set this to False, if being exactly deterministic is in need.
        torch.backends.cudnn.benchmark = True
