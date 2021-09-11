import os
import random
import argparse

parser = argparse.ArgumentParser(description='A multi-thread tool to crop sub images')
parser.add_argument('--train_path', type=str, default='../data/pixelshift200/train/train_rggb_512')
parser.add_argument('--test_path', type=str, default='../data/pixelshift200/test/test_rggb_1024')
args = parser.parse_args()

train_data_path = args.train_path
test_data_path = args.test_path

train_dst_path = 'train_pixelshift.txt'
val_dst_path = 'val_pixelshift.txt'
# val_datasets_num = 200
# type_list = ['png', 'PNG', 'tiff', 'tif', 'TIFF', 'JPG', 'jgp']
type_list = ['mat', 'MAT']
# remove old
if os.path.exists(train_dst_path):
    os.system('rm '+train_dst_path)
if os.path.exists(val_dst_path):
    os.system('rm '+val_dst_path)

# change to absolute path
if train_data_path[0] != '/':
    train_data_path = os.path.join(os.getcwd(), train_data_path)
if test_data_path[0] != '/':
    test_data_path = os.path.join(os.getcwd(), test_data_path)


# image list
lines = []
for file in os.listdir(train_data_path):
    if file.split('.')[-1] in type_list:
        lines.append(os.path.join(train_data_path, file)+'\n')
random.shuffle(lines)
train_lines = lines

lines = []
for file in os.listdir(test_data_path):
    if file.split('.')[-1] in type_list:
        lines.append(os.path.join(test_data_path,file)+'\n')
random.shuffle(lines)
val_lines = lines

# write datalist
with open(train_dst_path, 'a') as train_files:
    for line in train_lines:
        train_files.write(line)
with open(val_dst_path, 'w')as val_files:
    for line in val_lines:
        val_files.write(line)



