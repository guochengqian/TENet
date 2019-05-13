import os
import random
import pdb

# data_path = '/data/datasets/DIV2K_train_HR/gt'
data_path = '/mnt/lustre/share/qgc_datasets/DF2K-cropped-256/'
train_dst_path = 'train_df2k.txt'
valid_dst_path = 'valid_df2k.txt'
valid_datasets_num = 200
# type_list = ['png', 'PNG', 'tiff', 'tif', 'TIFF', 'JPG', 'jgp']
type_list = ['png', 'PNG']
# remove old
if os.path.exists(train_dst_path):
    os.system('rm '+train_dst_path)
if os.path.exists(valid_dst_path):
    os.system('rm '+valid_dst_path)

# image list
lines = []
for file in os.listdir(data_path):
    if file.split('.')[-1] in type_list:
        lines.append(os.path.join(data_path, file)+'\n')
random.shuffle(lines)

valid_lines = lines[0:valid_datasets_num]
train_lines = lines

# write datalist
with open(train_dst_path, 'a') as train_files:
    for line in train_lines:
        train_files.write(line)
with open(valid_dst_path, 'w')as valid_files:
    for line in valid_lines:
        valid_files.write(line)


