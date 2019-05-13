import os
import os.path
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from libtiff import TIFF

from TorchTools.progress_bar import ProgressBar
# from TorchTools.DataTools.FileTools import _pil2cv

def main():
    """A multi-thread tool to crop sub imags."""
    input_folder = '/data/sony/raw2019/demosaic'
    select_folder = '/data/sony/raw2019/patch'
    waste_folder = '/data/sony/raw2019/waste_patch'
    n_thread = 1
    crop_sz = 512
    stride = 256
    thres_sz = 100
    var_thresh = 20
    compression_level = 0  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(select_folder):
        os.makedirs(select_folder)
        print('mkdir [{:s}] ...'.format(select_folder))

    if not os.path.exists(waste_folder):
        os.makedirs(waste_folder)
        print('mkdir [{:s}] ...'.format(waste_folder))

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    # def update(arg):
    #     pbar.update(arg)
    #
    # pbar = ProgressBar(len(img_list))
    #
    # pool = Pool(n_thread)
    # for path in img_list:
    #     pool.apply_async(worker,
    #         args=(path, save_folder, crop_sz, stride, thres_sz, thres_sz, var_thresh),
    #         callback=update)
    # pool.close()
    # pool.join()
    for path in img_list:
        worker(path, select_folder, waste_folder, crop_sz, stride, thres_sz, var_thresh)
    print('All subprocesses done.')


def worker(path, select_folder, waste_folder, crop_sz, stride, thres_sz, var_thresh):
    img_name = os.path.basename(path)
    img = TIFF.open(path, mode='r')
    img = img.read_image()

    n_channels = len(img.size)
    if n_channels == 2:
        h, w = img.size
    elif n_channels == 3:
        h, w, c = img.size
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, stride)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, stride)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            patch_name = img_name.replace('.tiff', '_s{:05d}.tiff'.format(index))
            patch = img.crop([y, x, y + crop_sz, x + crop_sz])

            patch_np = np.asarray(patch)
            im_gray = cv2.cvtColor(patch_np, cv2.COLOR_BGR2GRAY)
            im_var = cv2.Laplacian(im_gray, cv2.CV_64F).var()
            if im_var < var_thresh:
                patch.save(os.path.join(select_folder, patch_name))
                print('saving: %s' % os.path.join(select_folder, patch_name))
            else:
                patch.save(os.path.join(waste_folder, patch_name))
                print('saving: %s' % os.path.join(waste_folder, patch_name))

    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
