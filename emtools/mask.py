# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
Masking module for EMTools package

@author: Andrew Herzing
"""

import numpy as np
import matplotlib.pylab as plt


def get_mask(s, r, r_outer=None, filter_type='circular'):
    """
    Create a 2D circular or annular logical mask.

    Args
    ----------
    s : Hyperspy Signal2D
        2D data to mask
    r : int
        Radius of the mask. If 'type' is circular, this defines the circle
        radius.  If 'annular', this defines the inner radius.
    r_outer : int
        For 'type' annular, defines the outer radius of the mask.
    filter_type : str
        Must be either 'circular' or 'annular'


    Returns
    ----------
    mask : Numpy array
        2D logical mask

    """
    h, w = s.data.shape
    center = [np.int32(s.data.shape[0] / 2), np.int32(s.data.shape[1] / 2)]

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    if filter_type == 'circular':
        mask = dist_from_center <= r
    elif filter_type == 'annular':
        mask = np.logical_and(dist_from_center <= r_outer,
                              dist_from_center > r)
    else:
        raise ValueError("Unknow mask type %s. Must be 'circle' or 'annular'."
                         % filter_type)
    return mask


def display_masks(im, masks, mask_alpha=0.5, log=True, im_min=8, im_max=11):
    """
    Overlay mask on image for visualization.

    Args
    ----------
    im : Hyperspy Signal2D
        Image or pattern to display
    masks : list
        Iterable of masks to overlay on image or pattern
    mask_alpha : float
        Degree of transparency of the masks.  Must be between 0.0 and 1.0
    log : bool
        If True, display image on a log scale (useful for diffraction patterns)
    im_min : float
        Minimum value for image display
    im_max : float
        Maximum value for image display

    Returns
    ----------

    """
    nmasks = len(masks)
    levels = np.linspace(2, 9, nmasks)
    total = np.zeros(masks[0].shape)
    for i, _ in enumerate(masks):
        total += levels[i] * masks[i]
    plt.figure()
    if log:
        plt.imshow(np.log(im + 1), vmin=im_min, vmax=im_max)
    else:
        plt.imshow(im, vmin=im_min, vmax=im.max)
    plt.imshow(total, cmap='nipy_spectral', vmin=0, vmax=10, alpha=mask_alpha)
    return
