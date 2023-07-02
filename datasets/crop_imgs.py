import os
import os.path as osp
import numpy as np
import cv2
from multiprocessing import Pool
import glob
import argparse


def main():
    parser = argparse.ArgumentParser(description='A multi-thread tool to crop sub images')
    parser.add_argument('--src_dir', type=str,
                        default='../data/DIV2K/DIV2K_train_HR',
                        help='path to original images folder')
    parser.add_argument('--save_dir', type=str,
                        default='../data/DIV2K/DIV2K_train_HR_sub',
                        help='path to output folder')
    parser.add_argument('--cont_var', type=float, default=0.,
                        help='content threshold for keeping or throwing a patch')
    parser.add_argument('--freq_var', type=float, default=0.,
                        help='frequency threshold for keeping or throwing a patch')
    parser.add_argument('--waste_dir', type=str,
                        default='',
                        help='path to put the wasted patches')
    parser.add_argument('--crop_sz', type=int, help='Crop size.',
                        default=480)
    parser.add_argument('--stride', type=int, help='stride for overlapped sliding window.',
                        default=240)
    parser.add_argument('--thres_sz', type=int,
                        help='Threshold size.Patches whose size is lower than thresh_size will be dropped.',
                        default=48)
    parser.add_argument('--n_thread', type=int, help='cpu_threads',
                        default=30)
    args = parser.parse_args()

    ext = 'png'
    crop_sz = args.crop_sz   # Crop size.
    stride = args.stride     # stride for overlapped sliding window.
    thres_sz = args.thres_sz   # Threshold size.Patches whose size is lower than thresh_size will be dropped.
    n_thread = args.n_thread    # cpu_threads

    enable_waste = (args.cont_var > 0) or (args.freq_var > 0)
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('mkdir [{:s}] ...'.format(args.save_dir))
    if enable_waste and not osp.exists(args.waste_dir):
        os.makedirs(args.waste_dir)
        print('mkdir [{:s}] ...'.format(args.waste_dir))

    # for DIV2K, 800
    # for DF2K, 800 (DIV2K) + 2650 (Flickr2k)
    img_list = sorted(
        glob.glob(osp.join(args.src_dir, '*' + ext))
    )
    print(f"find {len(img_list)} images in {args.src_dir} in total. ")
    pool = Pool(n_thread)
    for num, path in enumerate(img_list):
        print('processing {}/{}'.format(num, len(img_list)))
        pool.apply_async(worker, args=(path, args.save_dir,  args.waste_dir,
                                       crop_sz, stride, thres_sz, args.cont_var, args.freq_var))
    pool.close()
    pool.join()

    print('All subprocesses done.')


def worker(path, dst_folder, waste_img_folder, crop_sz, stride, thres_sz, cont_var_thresh, freq_var_thresh):
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w, c = img.shape

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
            patch_name = img_name.replace('.png', '_s{:05d}.png'.format(index))
            patch = img[x:x + crop_sz, y:y + crop_sz, :]

            im_gray = patch[:, :, 1]

            [mean, var] = cv2.meanStdDev(im_gray)
            freq_var = cv2.Laplacian(im_gray, cv2.CV_8U).var()

            if var > cont_var_thresh and freq_var > freq_var_thresh:
                cv2.imwrite(osp.join(dst_folder, patch_name), patch)
            else:
                cv2.imwrite(osp.join(waste_img_folder, patch_name), patch)
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
