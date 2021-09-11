import os
import numpy as np
import shutil


class TestArgs:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # datasets args
        parser.add_argument('--test_path', type=str,
                            default='/data/sony/datasets_backup/real_test/raw/online4_iso100.tiff',
                            help='path to test_images')
        parser.add_argument('--denoise', action='store_true',
                            help='denoise store_true')
        parser.add_argument('--sigma', default=2, type=int,
                            help='noise level GAWN')
        parser.add_argument('--save_path',
                            default='/data/sony/datasets_backup/real_test/sr_output', type=str,
                            help='save_path')
        parser.add_argument('--datatype', default='uint8', type=str,
                            help='uint8 or uint16')
        parser.add_argument('--shift_x', default=0, type=int,
                            help='shift pixel horizontally')
        parser.add_argument('--shift_y', default=0, type=int,
                            help='shift pixel vertically')
        parser.add_argument('--crop_scale', default=1, type=int,
                            help='shift pixel vertically')
        # model args
        parser.add_argument('--pretrain', default='', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--model', default='demo', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--norm_type', default=None, type=str,
                            help='dm_block_type(default: rrdb)')
        parser.add_argument('--block_type', default='rrdb', type=str,
                            help='dm_block_type(default: rrdb)')
        parser.add_argument('--act_type', default='relu', type=str,
                            help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--bias', action='store_true',
                            help='bias of layer')
        parser.add_argument('--channels', default=64, type=int,
                            help='channels')
        # for single task
        parser.add_argument('--n_blocks', default=6, type=int,
                            help='number of basic blocks')
        # for joint task
        parser.add_argument('--sr_n_blocks', default=6, type=int,
                            help='number of super-resolution blocks')
        parser.add_argument('--dm_n_blocks', default=6, type=int,
                            help='number of demosaicking blocks')
        # for super-resolution
        parser.add_argument('--scale', default=1, type=int,
                            help='Scale of Super-resolution.')

        parser.add_argument('--show_info',action='store_true',
                            help='print information')
        parser.add_argument('--postname',default='', type=str,
                            help='postname')

        self.args = parser.parse_args()

        self.args.noise_level = self.args.sigma/255.
        if self.args.datatype == 'uin16':
            self.args.datatype = np.uint16
        else:
            self.args.datatype = np.uint8

        if self.args.denoise:
            self.args.post = self.args.model+'_sigma'+str(self.args.sigma) +'_'+self.args.postname+ '.png'
        else:
            self.args.post = self.args.model+'_'+self.args.postname+'.png'
        return self.args

    def print_args(self):
        # print args
        print("==========================================")
        print("==========       CONFIG      =============")
        print("==========================================")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("\n")
        # check for folders existence 
        # if not os.path.exists(self.args.logdir):
        #     os.makedirs(self.args.logdir)
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        else:
            shutil.rmtree(self.args.save_path)



