import numpy
import math

import skimage

def psnr(img1, img2, PIXEL_MAX = 255.0):
    mse = numpy.mean( (img1 - img2) ** 2 )
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

