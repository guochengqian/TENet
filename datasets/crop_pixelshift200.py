import os
import os.path as osp
from glob import glob
import numpy as np
from scipy.io import loadmat, savemat
import argparse
from multiprocessing import Pool


def main():
    parser = argparse.ArgumentParser(description='A multi-thread tool to crop sub images')
    parser.add_argument('--src_path', type=str, default='../data/pixelshift200/train/train_rggb',
                        help='path to original mat folder')
    parser.add_argument('--save_path', type=str, default='../data/pixelshift200/train/train_rggb_512',
                        help='path to output folder')
    args = parser.parse_args()
    args.save_path = osp.join(os.getcwd(), args.save_path)

    os.makedirs(args.save_path, exist_ok=True)
    n_thread = 32
    crop_sz = 512
    stride = 512
    thres_sz = 256  # keep the regions in the last row/column whose thres_sz is over 256.
    ext = 'mat'

    img_list = sorted(
        glob(osp.join(os.getcwd(), args.src_path, '*' + ext))
    )

    # for num, path in enumerate(img_list):
    #     print('processing {}/{}'.format(num, len(img_list)))
    #     worker(path, args.save_path, crop_sz, stride, thres_sz)
    # print('All subprocesses done.')
    pool = Pool(n_thread)
    for num, path in enumerate(img_list):
        print('processing {}/{}'.format(num, len(img_list)))
        pool.apply_async(worker, args=(path, args.save_path, crop_sz, stride, thres_sz))
    pool.close()
    pool.join()

    print('All subprocesses done.')


def worker(path, save_dir, crop_sz, stride, thres_sz):
    raw_name = osp.basename(path)
    raw = loadmat(path)
    raw = np.asarray(raw['mat_crop'])

    h, w, c = raw.shape
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
            patch_name = raw_name.replace('.mat', '_s{:05d}.mat'.format(index))
            patch = raw[x:x + crop_sz, y:y + crop_sz, :]
            savemat(osp.join(save_dir, patch_name), {'raw': patch})


if __name__ == '__main__':
    main()
