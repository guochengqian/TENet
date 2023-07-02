"""
Author: Guocheng Qian, Yuanhao Wang
Contact: guocheng.qian@kaust.edu.sa

"""

import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import AnnotationBbox, AuxTransformBox
import imageio
import numpy as np
from PIL import Image
import cv2
import argparse
import glob

parser = argparse.ArgumentParser(description='Evaluation Data preparation')
parser.add_argument('--result_dir',
                    default='pretrain/',
                    help='path to the results (output images)')
parser.add_argument('--gt_dir',
                    default='data/benchmark/pixelshift/gt',
                    help='path to the GT images')
parser.add_argument('--save_dir',
                    default='pretrain/result')
parser.add_argument('--pipeline',
                    default='dn-sr-dm', type=str,
                    help='pipeline')
parser.add_argument('--scale',
                    default=2, type=int,
                    help='pipeline')
parser.add_argument('--pretrain_dataset',
                    default='pixelshift', type=str,
                    help='pipeline')
parser.add_argument('--dataset',
                    default='pixelshift', type=str,
                    help='pipeline')
parser.add_argument('--img_name',
                    default='003', type=str,
                    help='pipeline')
args = parser.parse_args()

scale = 4
region = [500, 500]
w, h = 100, 100

dpi = 300
linewidth = 4
edgecolor = 'r'

pipeline = f'{args.pipeline}-SR{args.scale}'
result_dir = osp.join(args.result_dir, args.pretrain_dataset, 'pipeline', f'result_{args.dataset}', pipeline, 'result')

img_list = glob.glob(osp.join(result_dir, f'*{args.img_name}*output*.png'))
assert len(img_list) == 1

img_path = img_list[0]
img_name = osp.basename(img_path)

image = Image.open(img_path)
sizes = np.shape(image)


region_img = np.array(image)[region[0]:region[0]+h, region[1]:region[1]+w, :]

# Now, resize the img
image_scaled = image.resize([sizes[0]//scale, sizes[1]//scale], Image.BILINEAR)
sizes_scaled = np.shape(image_scaled)

# show the image
fig = plt.figure(figsize=(sizes_scaled[0] / dpi, sizes_scaled[1] / dpi), dpi=dpi)
ax = plt.Axes(fig, [0., 0., 1., 1.])
fig.add_axes(ax)
ax.axis('off')


# now add bounding box to the cropped region
w_scaled, h_scaled = w/scale, h/scale
region_scaled = [region[0]//scale, region[1]//scale]
lw_scaled = linewidth/scale
r = patches.Rectangle(region_scaled, w_scaled, h_scaled, fc='none', ec='none', lw=0)
offsetbox = AuxTransformBox(ax.transData)
offsetbox.add_artist(r)
ab = AnnotationBbox(offsetbox,
                    (region_scaled[0] + w_scaled / 2., region_scaled[1] + h_scaled / 2.),
                    bboxprops=dict(facecolor="none", edgecolor='r', lw=lw_scaled))
ax.add_artist(ab)


# TODO: wrong here.
# now add bounding box to the upscaled region
xy = (linewidth, sizes_scaled[0] - h - linewidth-1)
r = patches.Rectangle(xy, w, h, fc='none', ec='none', lw=0)
offsetbox = AuxTransformBox(ax.transData)
offsetbox.add_artist(r)
ab = AnnotationBbox(offsetbox, (xy[0] + w / 2., xy[1] + w / 2.),
                    bboxprops=dict(facecolor="none", edgecolor='r', lw=linewidth))
ax.add_artist(ab)

#
# # replace
# image_scaled = np.array(image_scaled)
# image_scaled[linewidth: linewidth + region_size, sizes[0] - h - linewidth-1:sizes[0] - h - linewidth-1+region_size, :] = region_img

ax.imshow(image_scaled)


os.makedirs(args.save_dir, exist_ok=True)
output_path = osp.join(args.save_dir, img_name)
# os.remove(output_path)
plt.savefig(output_path, pad_inches=0)
print('test here')
