import os
import os.path
import sys
import numpy as np
import cv2
from scipy.io import loadmat, savemat
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def main():
    """A multi-thread tool to crop sub imags."""
    main_folder = '/data/pixel-shift-200/'
    input_folder = '/data/TENet/PixelShift200/PixelShift200_train'
    select_folder = os.path.join(main_folder, 'PixelShift200_train_crop')
    waste_folder = os.path.join(main_folder, 'waste_mat')
    img_folder = os.path.join(main_folder, 'img_crop')
    waste_img_folder = os.path.join(main_folder, 'waste_img')

    crop_sz = 512
    stride = 512
    thres_sz = 100
    cont_var_thresh = 0.20
    freq_var_thresh = 50

    if not os.path.exists(select_folder):
        os.makedirs(select_folder)
        print('mkdir [{:s}] ...'.format(select_folder))

    if not os.path.exists(waste_folder):
        os.makedirs(waste_folder)
        print('mkdir [{:s}] ...'.format(waste_folder))

    if not os.path.exists(img_folder):
        os.makedirs(img_folder)
        print('mkdir [{:s}] ...'.format(img_folder))

    if not os.path.exists(waste_img_folder):
        os.makedirs(waste_img_folder)
        print('mkdir [{:s}] ...'.format(waste_img_folder))

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    for num, path in enumerate(img_list):
        print('processing {}/{}'.format(num, len(img_list)))
        worker(path, select_folder, waste_folder, img_folder, waste_img_folder, crop_sz, stride, thres_sz, cont_var_thresh, freq_var_thresh)
    print('All subprocesses done.')


def worker(path, select_folder, waste_folder, img_folder, waste_img_folder, crop_sz, stride, thres_sz, cont_var_thresh, freq_var_thresh):
    img_name = os.path.basename(path)
    img = loadmat(path)
    img = np.asarray(img['ps4k'])

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
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

            patch_name = img_name.replace('.mat', '_s{:05d}.mat'.format(index))
            img_patch_name = img_name.replace('.mat', '_s{:05d}.tiff'.format(index))
            if n_channels == 2:
                patch = img[x:x + crop_sz, y:y + crop_sz]
            else:
                patch = img[x:x + crop_sz, y:y + crop_sz, :]

            # im_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            im_gray = patch[:, :, 1]

            [mean, var] = cv2.meanStdDev(im_gray)
            var = var/mean
            freq_var = cv2.Laplacian(im_gray, cv2.CV_16U).mean()
            if var > cont_var_thresh and freq_var>freq_var_thresh:
                savemat(os.path.join(select_folder, patch_name), {'ps': patch})
                img_patch = np.delete(patch, 2, 2).astype(float)/(2.**16)
                img_patch = img_patch ** (1/2.2) *255.
                img_patch = np.clip(img_patch, 0, 255)
                cv2.imwrite(os.path.join(img_folder, img_patch_name), np.uint8(img_patch))
                # print('saving: %s' % os.path.join(select_folder, patch_name))
            else:
                savemat(os.path.join(waste_folder, patch_name), {'ps': patch})
                # img_patch = np.delete(patch, 2, 2)
                img_patch = np.delete(patch, 2, 2).astype(float)/(2.**16)
                img_patch = img_patch ** (1/2.2) * 255.
                img_patch = np.uint8(np.clip(img_patch, 0, 255))
                cv2.imwrite(os.path.join(waste_img_folder, img_patch_name), np.uint8(img_patch))
                # print('saving: %s' % os.path.join(select_folder, patch_name))
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
