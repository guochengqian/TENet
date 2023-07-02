from PIL import Image
import os
import torch
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def cal_freqgain(gt, pred, ita = 1e-6):
    img_fft = fft2(pred)
    img_fft = fftshift(img_fft)
    img_power = np.power(np.absolute(img_fft), 2)

    img_ground_fft = fft2(gt)
    img_ground_fft = fftshift(img_ground_fft)
    img_ground_power = np.power(np.absolute(img_ground_fft), 2)

    spectra = np.log((img_power + ita) / (img_ground_power + ita))
    
    # spectra = np.clip(spectra, -2, 2)
    # spectra1 = np.ones_like(img_ground_power)
    # shapespec = spectra1.shape

    # spectra1[ int(0.05*shapespec[0]):-int(0.05*shapespec[0]), int(0.05*shapespec[0]):-int(0.05*shapespec[1]), :] = spectra[
    #                                                                               int(0.05*shapespec[0]):-int(0.05*shapespec[0]), int(0.05*shapespec[0]):-int(0.05*shapespec[1]), :]
    # TODO: ADD this 
    spectra[spectra < 0] = 0
    # return spectra1.mean()
    return abs(spectra).mean()