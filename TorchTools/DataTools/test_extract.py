import os
import os.path
import sys
import numpy as np
import cv2
from scipy.io import loadmat, savemat
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


img_path = '/data/datasets/RawSR20181001/raw2019/mat4k/train/waste_patch/pixelshift_5_1_s00107.mat'

img = loadmat(img_path)
img = np.asarray(img['ps'])

im_gray = img[:, :, 1]

[mean, var] = cv2.meanStdDev(im_gray)
cont_var = var / mean
freq_var = cv2.Laplacian(im_gray, cv2.CV_16U).var() / mean

freq_var