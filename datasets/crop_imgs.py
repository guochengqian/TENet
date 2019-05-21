import os
import os.path
import sys
import numpy as np
import cv2
from scipy.io import loadmat, savemat
from multiprocessing import Pool

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    """A multi-thread tool to crop sub imags."""
    main_folder = '/data/Df2k'
    input_folder = os.path.join(main_folder, 'DF2k')
    select_folder = os.path.join(main_folder, 'df2k_crop256')
    waste_img_folder = os.path.join(main_folder, 'waste_img')

    crop_sz = 256
    stride = 256
    thres_sz = 100
    n_thread = 5
    cont_var_thresh = 0
    freq_var_thresh = 0

    if not os.path.exists(select_folder):
        os.makedirs(select_folder)
        print('mkdir [{:s}] ...'.format(select_folder))

    if not os.path.exists(waste_img_folder):
        os.makedirs(waste_img_folder)
        print('mkdir [{:s}] ...'.format(waste_img_folder))

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    pool = Pool(n_thread)
    for num, path in enumerate(img_list):
        print('processing {}/{}'.format(num, len(img_list)))
        pool.apply_async(worker, args=(path, select_folder,  waste_img_folder, crop_sz, stride, thres_sz, cont_var_thresh, freq_var_thresh))
    pool.close()
    pool.join()

    print('All subprocesses done.')


def worker(path, select_folder, waste_img_folder, crop_sz, stride, thres_sz, cont_var_thresh, freq_var_thresh):
    img_name = os.path.basename(path)
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

            if var > cont_var_thresh and freq_var>freq_var_thresh:
                cv2.imwrite(os.path.join(select_folder, patch_name), patch)
            else:
                cv2.imwrite(os.path.join(waste_img_folder, patch_name), patch)
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
