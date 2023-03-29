# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
Alignment module for EMTools package.

@author: Andrew Herzing
"""


import numpy as np
import hyperspy.api as hs
from pystackreg import StackReg


def get_stackreg_shifts(stack):
    """
    Calculate alignment shifts for image stack using PyStackReg.

    Args
    ----------
    stack : Hyperspy Signal2D
        Image stack to align.

    Returns
    ----------
    sr_shifts : NumPy array
        Calculated alignment shifts.

    """
    sr = StackReg(StackReg.TRANSLATION)
    sr_shifts = sr.register_stack(stack.data)
    sr_shifts = np.array([sr_shifts[i][:-1, 2][::-1]
                          for i in range(0, len(sr_shifts))])
    return sr_shifts


def register_stack(stack):
    """
    Align stack of images using PyStackReg.

    Args
    ----------
    stack : Hyperspy Signal2D
        Image stack to align.

    Returns
    ----------
    reg : Hyperspy Signal2D
        Aligned stack.
    transforms : NumPy array
        Calculated alignment shifts.

    """
    reg = stack.deepcopy()
    sr = StackReg(StackReg.TRANSLATION)
    transforms = sr.register_stack(stack.data, axis=0, reference='previous', verbose=True)
    reg.data = sr.transform_stack(stack.data, transforms)
    return reg, transforms


def apply_hanning(image):
    """
    Apply a Hanning window to an image to remove artifacts in FFT.

    Args
    ----------
    image : Hyperspy Signal2D
        Image to taper.

    Returns
    ----------
    image : Hyperspy Signal2D
        Image after applying the Hanning window.

    """
    h = np.hanning(image.data.shape[0])
    ham2d = np.sqrt(np.outer(h, h))
    image.data = image.data * ham2d
    return image


def apply_taper(image, taper_percent):
    """
    Taper image edge pixels to remove artifacts in FFT.

    Args
    ----------
    image : Hyperspy Signal2D
        Image to taper.
    taper_percent : int
        Percent of image pixels to taper at the edge.

    Returns
    ----------
    image : Hyperspy Signal2D
        Image with tapered edge pixels.

    """
    width = np.int32(np.round(taper_percent / 100 * image.data.shape[0]))
    image.data = np.pad(image.data, pad_width=width, mode='linear_ramp')
    return image


def get_ps(s, crop=True, hanning=True, taper=False, taper_percent=3,
           crop_factor=3):
    """
    Return the Fourier power spectrum of an image.

    Args
    ----------
    s : Hyperspy Signal2D
        Image for which to calculate the power spectrum
    crop : bool
        If true, crop the power spectrum output to a factor set by crop_factor.
        Default is True.
    hannign : bool
        If true, apply a Hannign window to the image prior to calculating FFT.
        Default is True.
    taper : bool
        If True, taper the edge pixels of the image prior to calculating FFT.
        Default is False.
    taper_percent : int
        Percent of image pixels to taper at the edge. Default is 3.
    crop_factor : int
        Factor to crop the output power spectrum by. Default is 4.


    Returns
    ----------
    ps : Hyperspy Signal2D
        Power spectrum of the input image

    """
    image = s.deepcopy()
    if hanning:
        image = apply_hanning(image)
    if taper:
        image = apply_taper(image, taper_percent)

    ps = hs.signals.Signal2D(np.log(image.fft(shift=True).amplitude()))
    if crop:
        offset = ps.data.shape[0] / crop_factor
        center = ps.data.shape[0] / 2
        ps = ps.isig[center - offset:center + offset,
                     center - offset:center + offset]
    return ps
