# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:00:53 2018

@author: aherzing
"""
import matplotlib.pylab as plt
import numpy as np
from scipy.signal import correlate
import tqdm


def butter_bpf(data, Dl, Dh, n=1):
    """
    Function to apply a Butterworth bandpass filter to an image or stack.

    Args
    ----------
    data : Numpy array
        2D image or 3D stack of images
    Dl : integer
        Low-frequency cutoff of the filter
    Dh : integer
        High-frequency cutoff
    n : integer
        Sampling rate (default = 1)

    Returns
    ----------
    out : Numpy array
        Filtered image or stack of images
    """

    height, width = data.shape[-2:]
    [u, v] = np.meshgrid(np.arange(-np.floor(width/2),
                         np.floor(width/2)),
                         np.arange(-np.floor(height/2),
                         np.floor(height/2)))

    uv = u**2+v**2
    Duv = np.sqrt(uv)

    butter_lp_kernel = 1/(1+(Duv/Dl**(2*n)))
    butter_hp_kernel = 1/(1+(0.414*Dh/Duv**(2*n)))
    kernel = butter_lp_kernel * butter_hp_kernel

    out = np.zeros(data.shape, data.dtype)
    if len(data.shape) == 2:
        fftshift = np.fft.fftshift(np.fft.fft2(data))
        filtered = fftshift*kernel
        out = np.abs(np.fft.ifft2(filtered))
    elif len(data.shape) == 3:
        for i in tqdm.tqdm(range(0, data.shape[0])):
            fftshift = np.fft.fftshift(np.fft.fft2(data[i, :, :]))
            filtered = fftshift*kernel
            out[i, :, :] = np.float32(np.abs(np.fft.ifft2(filtered)))
    return(out)


def crossCorr(im1, im2):
    """
    Function to cross correlate to input images and display the result

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


def bandFilter(data, in_radius=9, out_radius=60):
    """
    Function to cross correlate to input images and display the result

    Args
    ----------
    data : Stack object
        Tilt series data
    in_radius : integer
        Inner pixel radius for the low-frequency side of the filter
    out_radius : integer
        Outer pixel radius for the high-frequency side of the filter

    Returns
    ----------
    out : Stack object
        Filtered copy of the inpuot stac
    """

    out = data.deepcopy()
    im = data.data

    nx, ny = im.shape[1:3]
    a, b = (nx/2), (ny/2)
    y, x = np.ogrid[-a:nx-a, -b:nx-b]
    mask = np.logical_xor(
                          x**2 + y**2 <= out_radius**2,
                          x**2 + y**2 < in_radius**2)

    imFreq = np.fft.fft2(im)
    # _ = np.fft.fftshift(np.abs(imFreq))
    imFilt = np.fft.fftshift(imFreq) * mask
    imNew = np.real(np.fft.ifft2(np.fft.ifftshift(imFilt)))

    out.data = np.float32(imNew)
    return(out)
