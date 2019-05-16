import os


class BaseArgs():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # datasets args
        parser.add_argument('--train_list', type=str,
                            default='datasets/train_df2k.txt',  help='path to train list')
        parser.add_argument('--valid_list', type=str,
                            default='datasets/valid_df2k.txt', help='path to valid list')

        parser.add_argument('--denoise', action='store_true',  help='denoise store_true')
        parser.add_argument('--max_noise', default=0.0748, type=float, help='noise_level')
        parser.add_argument('--min_noise', default=0.00, type=float, help='noise_level')

        parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size (default:8)')
        parser.add_argument('--patch_size', default=64, type=int, help='height of patch (default: 64)')

        parser.add_argument('--in_channels', default=1, type=int, help='in_channels')
        parser.add_argument('--gt_channels', default=1, type=int, help='gt_channels')
        parser.add_argument('--get2label', action='store_true',  help='denoise store_true')

        # train args
        parser.add_argument('--total_epochs', default=1000, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--total_iters', default=10000000, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--start_iters', default=0, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr_adjust_freq', default=200000, type=int,
                            help='decent the lr in epoch number')
        parser.add_argument('--lr', default=1e-4, type=float,
                            help='initial learning rate')

        # valid args
        parser.add_argument('--valid_freq', default=5000, type=int,
                            help='epoch interval for valid (default:5000)')
        # logger parse
        parser.add_argument('--print_freq', default=20, type=int,
                            help='print frequency (default: 100)')
        parser.add_argument('--postname', type=str, default='',
                            help='postname of save model')
        parser.add_argument('--save_path', type=str, default='',
                            help='path of save model')
        parser.add_argument('--save_freq', default=10000, type=int,
                            help='save ckpoints frequency')

        # model args
        parser.add_argument('--pretrained_model', default='', type=str,
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
        parser.add_argument('--n_resblocks', default=6, type=int,
                            help='number of basic blocks')
        # for joint task
        parser.add_argument('--sr_n_resblocks', default=6, type=int,
                            help='number of super-resolution blocks')
        parser.add_argument('--dm_n_resblocks', default=6, type=int,
                            help='number of demosaicking blocks')
        # for super-resolution
        parser.add_argument('--scale', default=2, type=int,
                            help='Scale of Super-resolution.')
        parser.add_argument('--downsampler', default='avg', type=str,
                            help='downsampler of Super-resolution.')
        # loss args
        parser.add_argument('--vgg_path', default='/mnt/lustre/qianguocheng/codefiles/vgg19.pth',
                            type=str, help='vgg loss pth location')
        parser.add_argument('--vgg_loss', type=str, default='l1', help="loss L1 or L2 ['l1', 'l2']")
        parser.add_argument('--vgg_layer', type=str, default='5', help='number of threads to prepare data.')
        parser.add_argument('--dm_lambda', type=float, default=0.5, help='dm loss lamda')
        parser.add_argument('--sr_lambda', type=float, default=0.5, help='sr loss lamda')
        parser.add_argument('--vgg_lambda', type=float, default=0, help='vgg loss lamda, set 0 if no vgg_loss')
        parser.add_argument('--remove', action='store_true', help='remove save_path')
        # for test jupyter
        self.args = parser.parse_args()
        # for test jupyter
        # self.args = parser.parse_args(args[0)

        if self.args.denoise:
            self.args.post = self.args.model + '-dn-'+ self.args.train_list.split('/')[-1].split('.')[0].split('_')[-1]+'x'\
                             + str(self.args.n_resblocks)+'-'+str(self.args.sr_n_resblocks)+'-'+str(self.args.dm_n_resblocks)\
                             +'-'+str(self.args.channels) + '-' + str(self.args.scale) + '-' + self.args.block_type
        else:
            self.args.post = self.args.model + '-'+ self.args.train_list.split('/')[-1].split('.')[0].split('_')[-1]+'x'\
                             + str(self.args.n_resblocks)+'-'+str(self.args.sr_n_resblocks)+'-'+str(self.args.dm_n_resblocks)\
                             +'-'+str(self.args.channels) + '-' + str(self.args.scale) + '-' + self.args.block_type

        if self.args.postname:
            self.args.post = self.args.post +'-' + self.args.postname
        self.args.save_path = 'checkpoints/checkpoints'+'-'+self.args.post
        self.args.logdir = 'logs/'+self.args.post

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


