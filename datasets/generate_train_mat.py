import os
import random
import pdb

# data_path = '/data/sony/datasets_backup/simulated_datasets/train_df2k/rgb_gt/'
train_data_path = '/data/pixel-shift-200/PixelShift200/PixelShift200_crop'
test_data_path = '/data/pixel-shift-200/PixelShift200/PixelShift200_valid'


train_dst_path = 'train_mat.txt'
valid_dst_path = 'valid_mat.txt'
# valid_datasets_num = 200
# type_list = ['png', 'PNG', 'tiff', 'tif', 'TIFF', 'JPG', 'jgp']
type_list = ['mat', 'MAT']
# remove old
if os.path.exists(train_dst_path):
    os.system('rm '+train_dst_path)
if os.path.exists(valid_dst_path):
    os.system('rm '+valid_dst_path)

# image list
lines = []
for file in os.listdir(train_data_path):
    if file.split('.')[-1] in type_list:
        lines.append(os.path.join(train_data_path,file)+'\n')
random.shuffle(lines)
train_lines = lines

lines = []
for file in os.listdir(test_data_path):
    if file.split('.')[-1] in type_list:
        lines.append(os.path.join(test_data_path,file)+'\n')
random.shuffle(lines)
valid_lines = lines

# write datalist
with open(train_dst_path, 'a') as train_files:
    for line in train_lines:
        train_files.write(line)
with open(valid_dst_path, 'w')as valid_files:
    for line in valid_lines:
        valid_files.write(line)



