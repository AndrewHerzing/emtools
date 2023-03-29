# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
Colors module for EMTools package.

@author: Andrew Herzing
"""

import numpy as np
import matplotlib as mpl

# Define several colormaps
thermal = mpl.colors.\
    LinearSegmentedColormap.from_list('gatan_colormap',
                                      ['black', 'blue', 'green', 'red',
                                       'yellow', 'white'],
                                      256, 1.0)
just_red = mpl.colors.\
    LinearSegmentedColormap.from_list('red_colormap',
                                      ['black', 'red'],
                                      256, 1.0)
just_green = mpl.colors.\
    LinearSegmentedColormap.from_list('green_colormap',
                                      ['black', 'green'],
                                      256, 1.0)
just_blue = mpl.colors.\
    LinearSegmentedColormap.from_list('blue_colormap',
                                      ['black', 'blue'],
                                      256, 1.0)


def normalize(image):
    """
    Normalize an image between 0 and 255.

    Args
    ----------
    image : Numpy array
        2-D or 3-D numpy array to be normalized

    Returns
    ----------
    output : Numpy array
        Normalized version of input

    """
    output = image - image.min()
    output = np.uint8(255 * output / output.max())
    return output


def rgboverlay(im1, im2=None, im3=None):
    """
    Create an RGB overlay using one or more images.

    Args
    ----------
    im1 : Numpy array
        2-D Numpy array containg an image
    im2 : Numpy array
        2-D Numpy array containg an image
    im3 : Numpy array
        2-D Numpy array containg an image

    Returns
    ----------
    rgb : Numpy array
        3-D array which can be interpreted as RGB image by Matplotlib

    """
    if len(np.shape(im1)) == 3:
        rgb = np.dstack((normalize(im1[:, :, 0]),
                         normalize(im1[:, :, 1]),
                         normalize(im1[:, :, 2])))
    else:
        rgb = np.dstack((normalize(im1),
                         normalize(im2),
                         normalize(im3)))
    return rgb


def gen_cmap(color, alpha=None, nbins=256):
    """
    Create a Matplotlib colormap ranging from black to a user-defined color.

    Args
    ----------
    color : string
        Matplotlib compatible color string (Ex. 'red', 'blue', 'r', 'b', etc.)
    alpha : float
        Degree of transparency (Default is None)
    nbins : int
        Number of bins for the color map (Default is 256)

    Returns
    ----------
    cmap : Matplotlib colormap
        Matplotlib colormap

    """
    if alpha:
        cmap = mpl.colors.\
            LinearSegmentedColormap.from_list('my_cmap', ['black', color],
                                              nbins)
        cmap._init()
        cmap._lut[:, -1] = np.linspace(0, alpha, cmap.N + 3)
    else:
        cmap = mpl.colors.\
            LinearSegmentedColormap.from_list('my_cmap', ['black', color],
                                              nbins)
    return cmap


def merge_color_channels(im_list, color_list=None, normalization='single',
                         return_all_channels=False):
    """
    Merge up to six gray scale images into a color composite.

    Parameters
    ----------
    im_list : list of Numpy arrays (images) to be merged
    color_list : list of strings
        Should be valid matplotlib color strings, such as 'red', 'green',
        'blue, etc.
    normalization : str
        Either 'single' or 'global'; controls whether the color scales are
        normalized on a per-image basis (``'single'``), or globally over all
        images (``'global'``)
    return_all_channels : bool
        If False, return just the RGB overlay array.  If True, return a
        a dictionary with the RGB overlay and each individual color channel.
        Default is False.

    Returns
    -------
    images : Numpy array or Dictionary
        A Signal1D containing the RGB overlay (return_all_channels is False,
        default behavior) or a dictionary with the RGB overlay and each
        individual color channel (if return_all_channels is True)

    """
    color_cycle = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'gray']
    if len(im_list) > 7:
        raise ValueError('List must be at most 7 images long')
    if color_list is None:
        color_list = color_cycle[0:len(im_list)]
    if not all(x in color_cycle for x in color_list):
        raise ValueError("Invalid color. Only red, green, blue, cyan, yellow "
                         "magenta and gray allowed")

    # shapes is (N, 2) numpy array containing the height and width of each
    # image
    shapes = np.asarray([im.data.shape[:2] for im in im_list])
    # Compare all rows of shapes to first row, and check if all rows in each
    # column are identical
    isequal = np.all(shapes == shapes[0, :], axis=0)
    # isequal should be [True, True], if it's not, raise an error:
    if not np.all(isequal):
        raise ValueError("All images must be the same shape to build "
                         "composite. Image shapes were: \n{}".format(shapes))
    height, width = shapes[0, :]

    images = dict()
    images['rgb'] = np.zeros([height, width, 3])
    for i in color_list:
        images[i] = np.zeros([height, width, 3])

    def _normalize_channels(im_list, color_list, method, iter_number):
        # determine normalization denominator
        if method == 'single':
            norm_value = im_list[iter_number].max()
        elif method == 'global':
            maxvals = np.zeros(len(im_list))
            for im_num, _ in enumerate(im_list):
                maxvals[im_num] = im_list[im_num].max()
            maxval = maxvals.max()
            norm_value = maxval
        else:
            raise ValueError("Unknown normalization method."
                             "Must be 'single' or 'global'.")

        if color_list[iter_number] == 'red':
            images['red'][:, :, 0] = \
                im_list[iter_number] / norm_value  # red channel
        elif color_list[iter_number] == 'green':
            images['green'][:, :, 1] = \
                im_list[iter_number] / norm_value  # green channel
        elif color_list[iter_number] == 'blue':
            images['blue'][:, :, 2] = \
                im_list[iter_number] / norm_value  # blue channel
        elif color_list[iter_number] == 'yellow':
            images['yellow'][:, :, 0] = \
                im_list[iter_number] / norm_value  # red channel +
            images['yellow'][:, :, 1] = \
                im_list[iter_number] / norm_value  # green channel
        elif color_list[iter_number] == 'magenta':
            images['magenta'][:, :, 0] = \
                im_list[iter_number] / norm_value  # red channel +
            images['magenta'][:, :, 2] = \
                im_list[iter_number] / norm_value  # blue channel
        elif color_list[iter_number] == 'cyan':
            images['cyan'][:, :, 1] = \
                im_list[iter_number] / norm_value  # green channel +
            images['cyan'][:, :, 2] = \
                im_list[iter_number] / norm_value  # blue channel
        elif color_list[iter_number] == 'gray':
            images['gray'][:, :, 0] = \
                im_list[iter_number] / norm_value / 2  # red channel +
            images['gray'][:, :, 1] = \
                im_list[iter_number] / norm_value / 2  # green channel +
            images['gray'][:, :, 2] = \
                im_list[iter_number] / norm_value / 2  # blue channel
        else:
            raise ValueError("Unknown color. Must be red, green, blue, "
                             "yellow, magenta, cyan, or gray.")

    for i in range(0, len(im_list)):
        _normalize_channels(im_list, color_list, normalization, i)

    # Scale RGB images between 0 and 255 and convert to int32 to avoid
    # matplotlib warning
    for i in color_list:
        images['rgb'] += images[i]

    images['rgb'] = np.uint8(images['rgb'] * (255 / images['rgb'].max()))

    if return_all_channels:
        for i in color_list:
            images[i] = np.uint8(images[i] * (255 / images[i].max()))
            return images
    else:
        return images['rgb']
