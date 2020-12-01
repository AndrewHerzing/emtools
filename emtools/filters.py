# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
Filters module for EMTools package

@author: Andrew Herzing
"""

import matplotlib.pylab as plt
import numpy as np
from scipy.signal import correlate
import tqdm


def butter_bpf(image, d_low, d_high, n=1):
    """
    Apply a Butterworth bandpass filter to an image or stack.

    Args
    ----------
    image : Hyperspy Signal2D or TomoStack
        2D image or 3D stack of images
    d_low : integer
        Low-frequency cutoff of the filter
    d_high : integer
        High-frequency cutoff
    n : integer
        Sampling rate (default = 1)

    Returns
    ----------
    out : Signal2D or TomoStack
        Filtered image or stack of images
    """
    out = image.deepcopy()
    data = image.data
    height, width = data.shape[-2:]
    [u, v] = np.meshgrid(np.arange(-np.floor(width / 2),
                         np.floor(width / 2)),
                         np.arange(-np.floor(height / 2),
                         np.floor(height / 2)))

    uv = u**2 + v**2
    d_uv = np.sqrt(uv)

    butter_lp_kernel = 1 / (1 + (d_uv / d_low**(2 * n)))
    butter_hp_kernel = 1 / (1 + (0.414 * d_high / d_uv**(2 * n)))
    kernel = butter_lp_kernel * butter_hp_kernel

    if len(data.shape) == 2:
        fftshift = np.fft.fftshift(np.fft.fft2(data))
        filtered = fftshift * kernel
        out.data = np.abs(np.fft.ifft2(filtered))
    elif len(data.shape) == 3:
        for i in tqdm.tqdm(range(0, data.shape[0])):
            fftshift = np.fft.fftshift(np.fft.fft2(data[i, :, :]))
            filtered = fftshift * kernel
            out.data[i, :, :] = np.float32(np.abs(np.fft.ifft2(filtered)))
    return out


def cross_corr(im1, im2):
    """
    Cross correlate to input images and display the result

    Args
    ----------
    im1 : numpy array
        2-D array containing image number 1
    im2 : numpy array
        2-D array containing image number 2

    """
    corr = correlate(im1, im2, mode='same')
    plt.imshow(corr, cmap='viridis')
    return


def band_filter(data, in_radius=9, out_radius=60):
    """
    Apply a band-pass filter to an image

    Args
    ----------
    data : Hyperspy Signal2D or TomoStack
        Image or image stack to filter
    in_radius : integer
        Inner pixel radius for the low-frequency side of the filter
    out_radius : integer
        Outer pixel radius for the high-frequency side of the filter

    Returns
    ----------
    out : Hyperspy Signal2D or TomoStack
        Filtered copy of the input data
    """

    out = data.deepcopy()
    im = data.data

    if len(im.shape) == 2:
        nx, ny = im.shape
        a, b = (nx / 2), (ny / 2)
        y, x = np.ogrid[-a:nx - a, -b:nx - b]
        mask = np.logical_xor(x**2 + y**2 <= out_radius**2,
                              x**2 + y**2 < in_radius**2)
        im_freq = np.fft.fft2(im)
        im_filt = np.fft.fftshift(im_freq) * mask
        out.data = np.float32(np.real(np.fft.ifft2(np.fft.ifftshift(im_filt))))
    elif len(im.shape) == 3:
        nx, ny = im.shape[1:3]
        a, b = (nx / 2), (ny / 2)
        y, x = np.ogrid[-a:nx - a, -b:nx - b]
        mask = np.logical_xor(x**2 + y**2 <= out_radius**2,
                              x**2 + y**2 < in_radius**2)
        for i in tqdm.tqdm(range(0, data.shape[0])):
            im_freq = np.fft.fft2(im[i, :, :])
            im_filt = np.fft.fftshift(im_freq) * mask
            out.data[i, :, :] =\
                np.float32(np.real(np.fft.ifft2(np.fft.ifftshift(im_filt))))
    return out
