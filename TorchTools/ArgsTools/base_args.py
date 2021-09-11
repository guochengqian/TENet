import os
import os.path as osp
import sys
import time
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
        parser.add_argument('--dataset', type=str, default=None,
                            help='path to train list')
        parser.add_argument('--train_list', type=str, default='datasets/train_pixelshift200.txt',
                            help='path to train list')
        parser.add_argument('--val_list', type=str, default='datasets/val_pixelshift200.txt',
                            help='path to val list')
        parser.add_argument('--benchmark_path', type=str,
                            default='data/benchmark/urban100/urban100_noisy_lr_raw_srgb_x2.pt',
                            help='path to benchmarking dataset')

        parser.add_argument('--patch_size', default=128, type=int,
                            help='width and height for a patch (default: 128); '
                                 'if performing joint DM and SR, then use 128.')
        parser.add_argument('--in_channels', default=3, type=int,
                            help='in_channels, RGB')
        parser.add_argument('--gt_channels', default=3, type=int,
                            help='gt_channels, RGB')
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

        # train args
        parser.add_argument('--imgs_per_gpu', default=64, type=int,
                            help='batch size per GPU (default:64)')
        parser.add_argument('--n_gpus', default=1, type=int,
                            help='number of GPUs (default:1)')
        parser.add_argument('--max_epochs', default=1000, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=5e-4, type=float,
                            help='initial learning rate')
        parser.add_argument('--lr_decay_step', default=100, type=int,
                            help='learning rate decay step')
        parser.add_argument('--gamma', default=0.5, type=float,
                            help='learning rate decay gamma')
        parser.add_argument('--lr_scheduler', default='cos', type=str,
                            help='learning rate decay scheduler (step | cos)')

        # logger parse
        parser.add_argument('--root_dir', type=str, default='log',
                            help='path for saving experiment files')
        parser.add_argument('--img_freq', default=10, type=int,
                            help='show images every xxx epochs(default: 10)')
        parser.add_argument('--print_freq', default=20, type=int,
                            help='show loss information every xxx iterations(default: 100)')
        parser.add_argument('--eval_freq', default=50, type=int,
                            help='perform evaluation every xxx epochs(default: 10)')
        parser.add_argument('--epoch_freq', default=500, type=int,
                            help='save milestone epoch every 500 epochs (default: 500)')
        parser.add_argument('--vis_eval', default=False, action='store_true',
                            help='generate evaluation result (images) and upload to tensorboard')
        parser.add_argument('--use_wandb', action='store_true',
                            help='set this to true if wana use wandb')

        # model args
        parser.add_argument('--model', default='tenet', type=str,
                            help='model type (default: tenet)')
        parser.add_argument('--norm_type', default=None, type=str,
                            help='normalization_type(default: do not use BN or IN)')
        parser.add_argument('--block_type', default='res', type=str,
                            help='dm_block(default: res). res/dudb/rrdb')
        parser.add_argument('--act_type', default='relu', type=str,
                            help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--no_bias', action='store_false', dest='bias',
                            help='do not use bias of layer')
        parser.add_argument('--channels', default=64, type=int,
                            help='channels')
        parser.add_argument('--n_blocks', default=6, type=int,
                            help='number of basic blocks')
        parser.add_argument('--mid_out', action='store_true',
                            help='activate middle output supervision')

        # for super-resolution
        parser.add_argument('--scale', default=2, type=int,
                            help='Scale of Super-resolution. Default: 2')
        parser.add_argument('--downsampler', default='bic', type=str,
                            help='downsampler of Super-resolution. Bicubic or average downsampling.  bic / avg')
        # loss args
        parser.add_argument('--mid_lambda', type=float, default=1.0, help='lamda for the middle stage supervision')
        parser.add_argument('--grad_clip', type=float, default=0.0, help='clip gradient. True if >0')
        parser.add_argument('--skip_threshold', type=float, default=5, help='skip the batch is the loss is too large')
        parser.add_argument('--loss_on_srgb', action='store_true',
                            help='calculate the loss function values on sRGB')

        # test args
        parser.add_argument('--save_dir', type=str, default=None,
                            help='path to save the test result')
        parser.add_argument('--test_data', type=str, default='data/dnd_2017',
                            help='path to the test data (dnd dataset)')
        parser.add_argument('--pretrain', default='', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--pretrain_other', default='', type=str,
                            help='path to pretrained of other pipeline')

        # log args
        parser.add_argument('--wandb_entity', default='', type=str,
                            help='the account name of your wandb')

        args = parser.parse_args()

        if args.dataset is None:
            args.dataset = args.train_list.split('_')[-1].split('.')[0]
        else:
            args.train_list = f'datasets/train_{args.dataset}.txt'
            args.val_list = f'datasets/val_{args.dataset}.txt'

        args.batch_size = args.imgs_per_gpu * args.n_gpus

        args.exp_prefix = '-'.join([args.in_type, args.mid_type, args.out_type,
                                    args.model, args.dataset, args.block_type,
                                    'n' + str(args.n_blocks)])
        args.jobname = '-'.join([args.exp_prefix,
                                 'SR' + str(args.scale),
                                 'C' + str(args.channels),
                                 'B' + str(args.batch_size),
                                 'Patch' + str(args.patch_size),
                                 'Epoch' + str(args.max_epochs)])

        if args.loss_on_srgb:
            args.jobname += '-loss_on_srgb'

        args.mid_type = None if args.mid_type == 'None' else args.mid_type
        self.args = args
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===> generate log dir
        if self.args.phase == 'train':
            # generate exp_dir when pretrained model does not exist, otherwise continue training using the pretrained
            if not self.args.pretrain:
                self._generate_exp_directory()
            else:
                self.args.exp_name = os.path.basename(os.path.dirname(os.path.dirname(self.args.pretrain)))
                self.args.exp_dir = os.path.dirname(os.path.dirname(self.args.pretrain))
                self.args.ckpt_dir = os.path.join(self.args.exp_dir, "checkpoint")

            # set some value to Training mode
            self.args.output_mid = True if self.args.mid_type is not None else False

        elif not self.args.phase == 'debug':
            self.args.exp_dir = os.path.dirname(args.pretrain)
            # self.args.res_dir = os.path.join(os.path.dirname(self.args.exp_dir), 'result')
            # pathlib.Path(self.args.res_dir).mkdir(parents=True, exist_ok=True)

        if not self.args.phase == 'debug':
            self._configure_logger()
            self._configure_wandb()
        self._print_args()
        self.set_seed(self.args.seed)

    def _generate_exp_directory(self):
        """
        Helper function to create checkpoint folder. We save
        model checkpoints using the provided model directory
        but we add a sub-folder for each separate experiment:
        """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        self.args.exp_name = '-'.join([self.args.jobname, timestamp, str(uuid.uuid4())])

        self.args.exp_dir = osp.join(self.args.root_dir, self.args.exp_name)
        self.args.ckpt_dir = osp.join(self.args.exp_dir, "checkpoint")
        self.args.code_dir = osp.join(self.args.exp_dir, "code")
        # self.args.res_dir = osp.join(self.args.exp_dir, "result")
        pathlib.Path(self.args.exp_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        # pathlib.Path(self.args.res_dir).mkdir(parents=True, exist_ok=True)
        # ===> save scripts
        shutil.copytree('model', osp.join(self.args.code_dir, 'model'))
        shutil.copytree('TorchTools', osp.join(self.args.code_dir, 'TorchTools'))

        if not self.args.save_dir:
            self.args.save_dir = osp.dirname(osp.dirname(self.args.pretrain))
        self.args.save_dir = osp.join(self.args.save_dir, "result")
        pathlib.Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)

    def _configure_logger(self):
        """
        Configure logger on given level. Logging will occur on standard
        output and in a log file saved in model_dir.
        """
        self.args.loglevel = "info"
        numeric_level = getattr(logging, self.args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: {}'.format(self.args.loglevel))

            # configure logger to display and save log data
        # log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
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
        logging.info("save log, checkpoint and code to: {}".format(self.args.exp_dir))

    def _configure_wandb(self):
        if self.args.use_wandb:
            self.args.wandb = edict()
            self.args.wandb.entitiy = self.args.wandb_entity
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
        torch.backends.cudnn.benchmark = True  # set this to False, if being exactly deterministic is in need.
