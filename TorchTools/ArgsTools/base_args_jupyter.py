import argparse
import os
import torch

class BaseArgs():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # datasets args

        parser.add_argument('--train_list', type=str,
                            default='/mnt/lustre/qianguocheng/codefiles/p1demosaickv1025/datasets/train_df2k.txt',
                            help='path to train list')
        parser.add_argument('--valid_list', type=str,
                            default='/mnt/lustre/qianguocheng/codefiles/p1demosaickv1025/datasets/valid_df2k.txt',
                            help='path to valid list')
        parser.add_argument('--denoise', default=False, type=bool,
                            help='denoise true or false')
        # train args
        parser.add_argument('--epochs', default=1500000, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--start_epoch', default=0, type=int,
                            help='manual epoch numner (useful on restart)')
        parser.add_argument('--step_epoch', default=200, type=int,
                            help='decent the lr in epoch number')
        parser.add_argument('-b', '--batch_size', default=8, type=int,
                            help='mini-batch size (default:8)')
        parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                            help='initial learning rate')
        parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                            help='weight decay (default: 1e-4)')
        parser.add_argument('--only_train_sr', default=False, type=bool,
                            help='only_train_sr (default: False)')
        parser.add_argument('--only_train_dm', default=False, type=bool,
                            help='only_train_sr (default: False)')

        # valid args
        parser.add_argument('--valid_interval', default=5, type=int,
                            help='epoch interval for valid (default:5)')
        # logger parse
        parser.add_argument('--print_freq', default=20, type=int,
                            help='print frequency (default: 100)')
        parser.add_argument('--post', type=str, default='jupyter',
                            help='postname of save model')
        parser.add_argument('--save_path', type=str, default='./output/checkpointsjupyter',
                            help='path of save model')
        parser.add_argument('--save_epoch', default=50, type=int,
                            help='save ckpoints frequency')
        parser.add_argument('--logdir', default='./output/log_jupyter', type=str,
                            help='log dir')
        # datasets parse
        parser.add_argument('--height', default=64, type=int,
                            help='height of patch (default: 128)')
        parser.add_argument('--width', default=64, type=int,
                            help='width of patch (default: 128)')
        parser.add_argument('--patch_num', default=1, type=int,
                            help='numbers of patches (default: 1)')

        # model args
        parser.add_argument('--pretrained_model', action='store_true',
                            help='path to pretrained model(default: none)')
        parser.add_argument('--model', default='ddsr_v3', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--norm_type', default=None, type=str,
                            help='dm_block_type(default: rrdb)')
        parser.add_argument('--block_type', default='res', type=str,
                            help='dm_block_type(default: res)')
        parser.add_argument('--scale', default=2, type=int,
                            help='Scale of Super-resolution.')
        parser.add_argument('--sr_n_resblocks', default=16, type=int,
                            help='Scale of Super-resolution.')
        parser.add_argument('--dm_n_resblock', default=12, type=int,
                            help='Scale of Super-resolution.')
        # loss args
        parser.add_argument('--vgg_path', default='/mnt/lustre/qianguocheng/codefiles/vgg19.pth',
                            type=str, help='vgg loss pth location')
        parser.add_argument('--vgg_loss', type=str, default='l1', help="loss L1 or L2 ['l1', 'l2']")
        parser.add_argument('--vgg_layer', type=str, default='5', help='number of threads to prepare data.')
        parser.add_argument('--dm_lambda', type=float, default=0.5, help='dm loss lamda')
        parser.add_argument('--sr_lambda', type=float, default=0.5, help='sr loss lamda')
        parser.add_argument('--vgg_lambda', type=float, default=0, help='vgg loss lamda, set 0 if no vgg_loss')

        self.args = parser.parse_args(args=[])
        self.initialized = True
        return self.args

    def print_args(self):
        # print args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("\n")
        print("==========     CONFIG END    =============")
        # check for folders existence 
        if os.path.exists(self.args.logdir):
            cmd = 'rm -rf ' + self.args.logdir
            os.system(cmd)
        os.makedirs(self.args.logdir)

        if not os.path.exists(self.args.save_path):		
            os.makedirs(self.args.save_path)
        assert os.path.exists(self.args.train_list), 'train_list {} not found'.format(self.args.train_list)
        assert os.path.exists(self.args.valid_list), 'valid_list {} not found'.format(self.args.valid_list)