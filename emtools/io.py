# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
IO module for EMTools package.

@author: Andrew Herzing
"""

from scipy.io import savemat, loadmat
import numpy as np
import hyperspy.api as hs
import matplotlib.animation as animation
import matplotlib.pylab as plt


def save_axsia(s, filename=None):
    """
    Save a Hyperspy signal in an AXSIA-ready Matlab file.

    Args
    ----------
    s : Hyperspy Signal2D
    filename : string
        Name for output .MAT file

    """
    if not filename:
        filename = 'matlab.mat'
    savedict = {}
    savedict['nrows'] = np.float64(s.data.shape[1])
    savedict['ncols'] = np.float64(s.data.shape[0])
    savedict['nchannels'] = np.float64(s.data.shape[2])
    savedict['resolution'] = np.float64(s.axes_manager[-1].scale)
    s.unfold()
    savedict['specdata'] = s.data.T

    savemat(filename, savedict, format='5')

    s.fold()

    return


def axsia_to_hspy(filename, calibration_signal=None, im_shape=None):
    """
    Load output of MVSA analysis from AXSIA software as a Hyperspy signal.

    Args
    ----------
    filename : string
        Name for AXSIA output file to load
    calibration_signal : Hyperspy Signal2D
        Signal from which to get calibration info
    im_shape : tuple
        Number of rows and columns of the original dataset.

    """
    axsia = {}

    axsia_in = loadmat(filename)
    if 'nrows' and 'ncols' in axsia_in.keys():
        nrows = int(axsia_in['nrows'][0][0])
        ncols = int(axsia_in['ncols'][0][0])
    elif calibration_signal:
        nrows = calibration_signal.data.shape[0]
        ncols = calibration_signal.data.shape[1]
    elif im_shape:
        nrows = im_shape[0]
        ncols = im_shape[1]
    else:
        raise ValueError(
            'SVD Decomposition requires definition of image shape')

    if 'npures' in axsia_in.keys():
        npures = int(axsia_in['npures'][0][0])
    else:
        npures = axsia_in['purespectra'].shape[1]
    if 'nchannels' in axsia_in.keys():
        nchannels = int(axsia_in['nchannels'][0][0])
    else:
        nchannels = axsia_in['purespectra'].shape[0]
    loadings = axsia_in['concentrations']
    factors = axsia_in['purespectra']
    # method = axsia_in['SIparams'][0][0][6][0]

    axsia['loadings'] = \
        hs.signals.Signal2D(loadings.reshape([nrows, ncols, npures]))
    axsia['loadings'] = axsia['loadings'].swap_axes(0, 1).swap_axes(1, 2)

    axsia['factors'] = \
        hs.signals.Signal1D(factors.reshape([nchannels, npures]))
    axsia['factors'] = axsia['factors'].swap_axes(0, 1)

    if calibration_signal:
        axsia['factors'].axes_manager[1].name = \
            calibration_signal.axes_manager[-1].name
        axsia['factors'].axes_manager[1].offset = \
            calibration_signal.axes_manager[-1].offset
        axsia['factors'].axes_manager[1].scale = \
            calibration_signal.axes_manager[-1].scale
        axsia['factors'].axes_manager[1].units = \
            calibration_signal.axes_manager[-1].units

        axsia['loadings'].axes_manager[1].name = \
            calibration_signal.axes_manager[0].name
        axsia['loadings'].axes_manager[1].offset = \
            calibration_signal.axes_manager[0].offset
        axsia['loadings'].axes_manager[1].scale = \
            calibration_signal.axes_manager[0].scale
        axsia['loadings'].axes_manager[1].units = \
            calibration_signal.axes_manager[0].units

        axsia['loadings'].axes_manager[2].name = \
            calibration_signal.axes_manager[1].name
        axsia['loadings'].axes_manager[2].offset = \
            calibration_signal.axes_manager[1].offset
        axsia['loadings'].axes_manager[2].scale = \
            calibration_signal.axes_manager[1].scale
        axsia['loadings'].axes_manager[2].units = \
            calibration_signal.axes_manager[1].units

    return axsia


def savemovie(stack, start=0, stop=None, outfile='output.avi', fps=15, dpi=100, title='Series', clim=None, cmap='gray'):
    """
    Save the image series as an AVI movie file.

    Args
    ----------
    stack : Hyperspy Signal2D
        Image stack to save as movie
    start : integer
        Filename for output. If None, a UI will prompt for a filename.
    stop : integer
        Filename for output. If None, a UI will prompt for a filename.
    outfile : string
        Filename for output.
    fps : integer
        Number of frames per second at which to create the movie.
    dpi : integer
        Resolution to save the images in the movie.
    title : string
        Title to add at the top of the movie
    clim : tuple
        Upper and lower contrast limit to use for movie
    cmap : string
        Matplotlib colormap to use for movie

    """
    if clim is None:
        clim = [stack.data.min(), stack.data.max()]

    if stop is None:
        stop = stack.data.shape[0]

    fig, ax = plt.subplots(1, figsize=(8, 8))

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if title:
        ax.set_title(title)

    im = ax.imshow(stack.data[start, :, :], interpolation='none',
                   cmap=cmap, clim=clim)
    fig.tight_layout()

    def update_frame(n):
        tmp = stack.data[n, :, :]
        im.set_data(tmp)
        return im

    frames = np.arange(start, stop, 1)

    ani = animation.FuncAnimation(fig, update_frame, frames)

    writer = animation.writers['ffmpeg'](fps=fps)
    ani.save(outfile, writer=writer, dpi=dpi)
    plt.close()
    return
