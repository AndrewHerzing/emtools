# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
RDF module for EMTools package

@author: Andrew Herzing
"""

import numpy as np
from hyperspy.signals import Signal2D, Signal1D
from hyperspy.drawing.utils import plot_spectra
import matplotlib.pylab as plt


def data_out(xaxis, data, filename):
    """
    Save the calculated RDF to a text file.

    Args
    ----------
    xaxis : Numpy array
        Spatial frequency axis for the RDF
    data : Numpy array
        RDF results
    filename : str

    """
    out = np.array([xaxis, data])
    out = out.T
    with open(filename, 'wb') as f:
        np.savetxt(f, out, delimiter=' , ', fmt='%2e')
    f.close()
    return


def azimuthal_average(image, bin_size=0.5):
    """
    Calculate the azimuthal average 2-D array.

    Args
    ----------
    image : Numpy array
        Data on which to perform the azimuthal average. Typically a diffraction
        pattern or a power spectrum (2D FFT)
    bin_size : float
        Size of the bins to distribute the azimuthally averaged data

    Returns
    ----------
    bin_centers : Numpy array
        Location of the bin_centers of the azimuthal average.  Defines the
        x-axis of the RDF.
    radial_prof : Numpy array
        Azimuthally averaged radial profile of the input data.

    """
    y, x = np.indices(image.shape)
    center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])
    nbins = int(np.round(r.max() / bin_size) + 1)
    maxbin = nbins * bin_size
    bins = np.linspace(0, maxbin, nbins + 1)
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    radial_prof = np.histogram(r, bins, weights=(image))[0] / \
        np.histogram(r, bins, weights=np.ones(np.shape(image)))[0]
    return bin_centers, radial_prof


def RDF(s, rebin_factor=None, bin_size=1.0):
    """
    Calculate the radial profile the FFT of an image.

    Args
    ----------
    s : Hyperspy Signal2D
        Image from which to calculate the RDF
    rebin_factor : int
        If defined, the image is downsampled prior to analysis by this
        factor.
    bin_size : float
        Size of the bins to distribute the azimuthally averaged data. Default
        is 1.0.

    Returns
    ----------
    psd : Numpy array
        Power spectrum of the input image (i.e. the square of the real part of
        the 2D FFT)
    profile : Hyperspy Signal1D
        Radially average of the PSD

    """
    if rebin_factor:
        image = s.rebin(scale=(rebin_factor, rebin_factor))
    else:
        image = s.deepcopy()
    fft = image.fft(shift=True)
    psd = Signal2D(np.abs(fft.real())**2,
                   axes=[fft.axes_manager[0].get_axis_dictionary(),
                         fft.axes_manager[1].get_axis_dictionary()])
    xaxis, profile = azimuthal_average(psd.data, bin_size)
    scale = fft.axes_manager[0].scale
    profile = Signal1D(profile)
    profile.axes_manager[0].name = 'Frequency'
    profile.axes_manager[0].units = '1/%s' % s.axes_manager[0].units
    profile.axes_manager[0].offset = xaxis[0] * scale
    profile.axes_manager[0].scale = scale
    return psd, profile


def plotRDF(rdfs, xrange=None, yrange=None, labels=None):
    """
    Plot a calculated RDF or series of RDFs on a log-log scale.

    Args
    ----------
    rdfs : Hyperspy Signal2D or list
        RDF data to plot
    xrange : tuple
        Minimum and maximum value at which to truncate the plot in the
        horizontal axis.
    yrange : tuple
        Minimum and maximum value at which to truncate the plot in the
        vertical axis.
    labels : str or list
        Labels for each RDF to use in the plot legend.

    Returns
    ----------
    fig : Matplotlib Figure
    ax : Matplotlib Axis

    """
    if len(rdfs) == 1:
        rdfs.plot()
        ax = plt.gca()
        fig = plt.gcf()
        ax.set_yscale("log")
        ax.set_xscale("log")
        line = ax.get_lines()[0]
        line.set_linewidth(0)
        line.set_marker('o')
        line.set_markeredgecolor('red')
        line.set_markerfacecolor('white')
        if xrange:
            ax.set_xlim(xrange)
        if yrange:
            ax.set_ylim(yrange)
    else:
        ax = plot_spectra(rdfs, legend=labels)
        fig = plt.gcf()
        ax.set_yscale("log")
        ax.set_xscale("log")
        lines = ax.get_lines()
        for line in lines:
            line.set_linewidth(0)
            line.set_marker('o')
            line.set_markerfacecolor('white')
        if xrange:
            ax.set_xlim(xrange)
        if yrange:
            ax.set_ylim(yrange)
    return fig, ax
