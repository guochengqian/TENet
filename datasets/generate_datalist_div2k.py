import os
import os.path as osp
import argparse
import glob
import random

parser = argparse.ArgumentParser(description='A multi-thread tool to crop sub images')
parser.add_argument('--train_dir', type=str,
                    default='../data/DIV2K/DIV2K_train_HR_sub',
                    help='path to training images')
parser.add_argument('--val_dir', type=str,
                    default='../data/DIV2K/DIV2K_val5_HR_sub',
                    help='path to val images')
args = parser.parse_args()

train_dst_path = 'train_div2k.txt'
val_dst_path = 'val_div2k.txt'
ext = 'png'
type_list = ['png', 'PNG', 'tiff', 'tif', 'TIFF', 'JPG', 'jgp', 'bmp', 'BMP']

# remove old
if os.path.exists(train_dst_path):
    os.system('rm '+train_dst_path)
if os.path.exists(val_dst_path):
    os.system('rm '+val_dst_path)

# change to absolute path
if args.train_dir[0] != '/':
    args.train_dir = osp.join(os.getcwd(), args.train_dir)
if args.val_dir[0] != '/':
    args.val_dir = osp.join(os.getcwd(), args.val_dir)

# image list
train_lines = sorted(
    glob.glob(osp.join(args.train_dir, '*' + ext))
)
random.seed(0)
random.shuffle(train_lines)


val_lines = sorted(
    glob.glob(osp.join(args.val_dir, '*' + ext))
)

# write datalist
with open(train_dst_path, 'a') as train_files:
    for line in train_lines:
        train_files.write(line + '\n')
with open(val_dst_path, 'a')as val_files:
    for line in val_lines:
        val_files.write(line + '\n')


