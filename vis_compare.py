"""
Author: Guocheng Qian
Contact: guocheng.qian@kaust.edu.sa

"""

import matplotlib.pyplot as plot
import cv2


img_path_list = [
    # "/home/qiang/codefiles/low_level/ISP/ispnet/data/testdata/result/APC_0022_3c590fa3578840848200694e78fa919d-noisy_lr_raw-raw-linrgb-res.png",
    # "/home/qiang/codefiles/low_level/ISP/ispnet/data/testdata/result/APC_0022_3c590fa3578840848200694e78fa919d-noisy_lr_raw-raw-linrgb-eam.png",
    "/home/qiang/codefiles/low_level/ISP/ispnet/data/testdata/result/APC_0022_3c590fa3578840848200694e78fa919d-noisy_lr_raw-raw-linrgb-rrdb.png"
]

crop_size = [100, 100]
crop_start = [3400, 5500]

imgs = []
for img_path in img_path_list:
    im = cv2.imread(img_path)[crop_start[0]: crop_start[0]+crop_size[0],crop_start[1]:crop_start[1]+crop_size[1]]
    cv2.imwrite(img_path.split('.')[0]+'_crop.png', im)

