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


class BaseArgs:
    def __init__(self, parser):
        parser.add_argument('--phase', type=str, default='train',
                            help='phase. Default: train')
        parser.add_argument('--seed', type=int, default=0)

        # datasets args
        parser.add_argument('--train_list', type=str, default='datasets/train_mirflickr.txt',
                            help='path to train list')
        parser.add_argument('--val_list', type=str, default='datasets/val_mirflickr.txt',
                            help='path to val list')
        parser.add_argument('--benchmark_path', type=str,
                            default='data/benchmark/urban100/urban100_noisy_lr_raw-srgb.pt',
                            help='path to val list')

        parser.add_argument('--patch_size', default=64, type=int,
                            help='width and height for a patch (default: 256)')
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

        # noise
        # parser.add_argument('--denoise', action='store_true', help='denoise store_true, using shot and read noise')
        parser.add_argument('--read_noise', default=0.00, type=float, help='read_noise')
        parser.add_argument('--shot_noise', default=0.00, type=float, help='shot_noise')

        # train args
        parser.add_argument('--batch_per_gpu', default=16, type=int,
                            help='batch size per GPU (default:16)')
        parser.add_argument('--n_gpus', default=1, type=int,
                            help='number of GPUs (default:1)')
        parser.add_argument('--max_epochs', default=500, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=1e-4, type=float,
                            help='initial learning rate')
        parser.add_argument('--lr_decay_step', default=50, type=int,
                            help='learning rate decay step')
        parser.add_argument('--gamma', default=0.5, type=float,
                            help='learning rate decay gamma')

        # logger parse
        parser.add_argument('--root_dir', type=str, default='log',
                            help='path for saving experiment files')
        parser.add_argument('--img_freq', default=10, type=int,
                            help='show images every xxx epochs(default: 10)')
        parser.add_argument('--print_freq', default=100, type=int,
                            help='show images every xxx iterations(default: 100)')
        # model args
        parser.add_argument('--model', default='tenet', type=str,
                            help='path to pretrained model (default: tenet)')
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
        parser.add_argument('--n_blocks', default=18, type=int,
                            help='number of basic blocks')
        parser.add_argument('--mid_out', action='store_true',
                            help='activate middle output supervision')
        parser.add_argument('--output_mid', action='store_true',
                            help='output the middle stage result')

        # for super-resolution
        parser.add_argument('--scale', default=2, type=int,
                            help='Scale of Super-resolution. Default: 2')
        parser.add_argument('--downsampler', default='bic', type=str,
                            help='downsampler of Super-resolution. Bicubic or average downsampling.  bic / avg')
        # loss args
        parser.add_argument('--mid_lambda', type=float, default=1.0, help='lamda for the middle stage supervision')
        parser.add_argument('--loss_on_srgb', action='store_true',
                            help='calculate the loss function values on sRGB')

        # test args
        parser.add_argument('--save_dir', type=str, default=None,
                            help='path to save the test result')
        parser.add_argument('--test_data', type=str, default='/home/wangy0k/Desktop/denoise/darmstadt/data/',
                            help='path to the test data (dnd dataset)')
        parser.add_argument('--pretrain', default='', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--pretrain', default='', type=str,
                            help='path to pretrained model(default: none)')
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

        args = parser.parse_args()

        args.dataset = args.train_list.split('_')[-1].split('.')[0]
        args.batch_size = args.batch_per_gpu * args.n_gpus

        args.pre_jobname = '-'.join([args.pre_in_type, args.mid_type, args.pre_out_type])
        args.jobname = '-'.join([args.in_type, args.mid_type, args.out_type])

        args.mid_type = None if args.mid_type == 'None' else args.mid_type
        self.args = args

        # ===> generate log dir
        self._generate_exp_directory()
        if self.args.phase == 'train':
            self._configure_logger()
        self._print_args()
        self.set_seed(self.args.seed)
        self.args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _generate_exp_directory(self):
        """
        Helper function to create checkpoint folder. We save
        model checkpoints using the provided model directory
        but we add a sub-folder for each separate experiment:
        """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        experiment_string = '_'.join([self.args.jobname, timestamp, str(uuid.uuid4())])

        if self.args.phase == 'train':
            self.args.exp_dir = osp.join(self.args.root_dir, experiment_string)
            self.args.ckpt_dir = osp.join(self.args.exp_dir, "checkpoint")
            self.args.code_dir = osp.join(self.args.exp_dir, "code")
            self.args.res_dir = osp.join(self.args.exp_dir, "result")
            pathlib.Path(self.args.exp_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.args.ckpt_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.args.res_dir).mkdir(parents=True, exist_ok=True)
            # ===> save scripts
            shutil.copytree('model', osp.join(self.args.code_dir, 'model'))
            shutil.copytree('TorchTools', osp.join(self.args.code_dir, 'TorchTools'))
        else:

            if not self.args.save_dir:
                self.args.save_dir = osp.dirname(self.args.pretrain)
            # self.args.save_dir = osp.join(self.args.save_dir, "result-{}".format(osp.basename(self.args.pretrain)))
            pathlib.Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)

    def _configure_logger(self):
        """
        Configure logger on given level. Logging will occur on standard
        output and in a log file saved in model_dir.
        """
        self.args.loglevel = "info"
        numeric_level = getattr(logging, self.args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: {}'.format(self.args.loglevelloglevel))

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
        torch.backends.cudnn.benchmark = False
