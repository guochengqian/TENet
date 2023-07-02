import os
import os.path as osp
import numpy as np
import cv2
from multiprocessing import Pool
import glob
import argparse
import shutil
import pathlib


def main():
    parser = argparse.ArgumentParser(description='A multi-thread tool to crop sub images')
    parser.add_argument('--src_path', type=str, default='/data/image/ispnet/mirflickr25k/original_dataset',
                        help='path to original images folder')
    parser.add_argument('--save_path', type=str, default='/data/image/ispnet/mirflickr25k/',
                        help='path to output folder')
    args = parser.parse_args()

    train_path = osp.join(args.save_path, 'mirflicker25k', 'train')
    test_path = osp.join(args.save_path, 'mirflicker25k', 'test')
    down_train_folder = osp.join(args.save_path, 'mirflicker25k_downsampled', 'train')
    down_test_folder = osp.join(args.save_path, 'mirflicker25k_downsampled', 'test')

    waste_img_folder = osp.join(args.save_path, 'waste_img')
    pathlib.Path(train_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(test_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(down_train_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(down_test_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(waste_img_folder).mkdir(parents=True, exist_ok=True)

    ext = 'jpg'
    down_scale = 2

    img_size = 256
    n_thread = 10
    train_ratio = 0.95

    img_list = sorted(
        glob.glob(osp.join(args.src_path, '*' + ext))
    )
    train_img_list = img_list[0:int(len(img_list)*train_ratio)]
    test_img_list = img_list[int(len(img_list)*train_ratio):]

    pool = Pool(n_thread)
    for num, path in enumerate(train_img_list):
        print('processing {}/{}'.format(num, len(train_img_list)))
        pool.apply_async(worker, args=(path, train_path,  down_train_folder,
                                       waste_img_folder, img_size, down_scale))
    pool.close()
    pool.join()

    pool = Pool(n_thread)
    for num, path in enumerate(test_img_list):
        print('processing {}/{}'.format(num, len(test_img_list)))
        pool.apply_async(worker, args=(path, test_path,  down_test_folder,
                                       waste_img_folder, 0, down_scale))
    pool.close()
    pool.join()


def worker(path, save_folder, down_folder, waste_img_folder, img_size, scale):
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w, c = img.shape
    lr_img = cv2.resize(img.copy(), (0, 0), fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)

    if h>=img_size and w>=img_size:
        shutil.copy(path, osp.join(save_folder, img_name))
        cv2.imwrite(osp.join(down_folder, img_name), lr_img)
    else:
        shutil.copy(path, osp.join(waste_img_folder, img_name))


if __name__ == '__main__':
    main()
