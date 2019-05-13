import cv2
from PIL import Image
import os
from multiprocessing import Pool
from TorchTools.DataTools.FileTools import _all_images
from TorchTools.Functions.functional import to_cv_array


def stride_crop_img(im_name):

    ## Image, Patch Size, Stride Preparation Based on HR/LR
    var_thresh = 20
    crop_edge = 10
    im = Image.open(im_name)
    im_name = os.path.basename(im_name).split('.')[0]
    w, h = im.size
    scale = 4 if is_hr else 1
    im = im.crop((crop_edge * scale, crop_edge * scale, w - crop_edge * scale, h - crop_edge * scale))
    patch_size = scale * lr_patch_size
    stride = scale * lr_stride
    if is_hr:
        patch_folder = os.path.join(dst_folder, 'HR_image')
        select_folder = os.path.join(dst_folder, 'val_HR')
    else:
        patch_folder = os.path.join(dst_folder, 'LR_image')
        select_folder = os.path.join(dst_folder, 'val_LR')

    cnt = 0
    # for x in range((w - patch_size) // stride + 1):
    #     for y in range((h - patch_size) // stride + 1):
    #         startx = x * stride
    #         starty = y * stride
    #         patch = im.crop([startx, starty, startx + patch_size, starty + patch_size])
    for x in range((h - patch_size) // stride + 1):
        for y in range((w - patch_size) // stride + 1):
            startx = x * stride
            starty = y * stride
            patch = im.crop([starty, startx, starty + patch_size, startx + patch_size])
            patch_name = '%s_%d.png' % (im_name, cnt)
            cnt += 1

            if is_hr:
                patch_cv = to_cv_array(patch)
                im_gray = cv2.cvtColor(patch_cv, cv2.COLOR_BGR2GRAY)
                im_var = cv2.Laplacian(im_gray, cv2.CV_64F).var()
                if im_var < var_thresh:
                    patch.save(os.path.join(select_folder, patch_name))
                    print('saving: %s' % os.path.join(select_folder, patch_name))
                else:
                    patch.save(os.path.join(patch_folder, patch_name))
                    print('saving: %s' % os.path.join(patch_folder, patch_name))
            else:
                patch.save(os.path.join(patch_folder, patch_name))
                print('saving: %s' % os.path.join(patch_folder, patch_name))


if __name__ == '__main__':

    global dst_folder
    global lr_patch_size
    global lr_stride
    global scale

    ## Parameter Settings
    data_dir = './'     # Pair Data Path, should have train_HR and rect_LR(sift cropped and warped LR)
    dst_folder = './train_patch/' # 裁剪结果文件夹
    lr_patch_size = 200
    lr_stride = 150
    scale = 4

    ## Folder Preparation
    hr_dir = os.path.join(data_dir, 'train_HR') # HR LR文件夹
    lr_dir = os.path.join(data_dir, 'rect_LR')
    global is_hr

    def makedir(path):
        path = os.path.join(dst_folder, path)
        if not os.path.exists(path):
            os.makedirs(path)
    makedir('HR_image')
    makedir('LR_image')
    makedir('val_HR')   # 根据方差筛选出的HR patch
    makedir('val_LR')   # LR patch先不筛选，人工再次筛选HR Patch后，根据HR的结果筛LR

    ## Stride Crop HR Patches
    print('Cropping HR Images...')
    hr_names = _all_images(hr_dir)
    is_hr = True
    pool = Pool()
    pool.map(stride_crop_img, hr_names)
    pool.close()
    pool.join()

    ## Stride Crop LR Patches
    print('Cropping LR Images...')
    lr_names = _all_images(lr_dir)
    is_hr = False
    pool = Pool()
    pool.map(stride_crop_img, lr_names)
    pool.close()
    pool.join()
