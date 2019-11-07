# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
EDS module for EMTools package

@author: Andrew Herzing
"""

from hyperspy.misc import elements
from matplotlib import pylab as plt
import hyperspy.api as hs
from skimage import measure, filters
import numpy as np


def plot_EDS(spec, axis=None, peaklabels=None, line_color='red',
             energy_range=None, intensity_range=None, horz_offset=None,
             vert_offset=None, font_size=8):
    """
    Plot several EDS spectra.

    Args
    ----------
    spec : Hyperspy Signal1D
        Single EDS spectrum signal
    axis : Matplotlib axis
        Axis in which to plot the data.  If None, a new Figure and Axis are
        created
    peak_labels : bool or list
        If True, label the peaks defined in spec.metadata.Sample.xray_lines.
        If list, the listed peaks are labeled.
    line_color : string
        Color for the spectral plots
    energy_range : tuple
        Plot is truncated horizonatally to the minimum and maximum value
    intensity_range : tuple
        Plot is truncated vertically to the minimum and maximum value
    horz_offset : float
        Offset from peak location (in calibrated values) with which to offset
        the labels in the horizontal direction
    vert_offset : float
        Offset from peak location (in calibrated values) with which to offset
        the labels in the vertical direction
    font_size : int
        Fontsize for labels


    Returns
    ----------
    figure : Matplotlib Figure instance
    axis : Matplotlib Axis instance

    """
    if axis is None:
        figure, axis = plt.subplots(1)
        out = True
    else:
        out = False
    axis.plot(spec.axes_manager[-1].axis, spec.data, color=line_color)
    if energy_range:
        axis.set_xlim(energy_range[0], energy_range[1])
    if intensity_range:
        axis.set_ylim(intensity_range[0], intensity_range[1])
    if peaklabels:
        if peaklabels is True:
            peaklabels = spec.metadata.Sample.xray_lines
        elif type(peaklabels) is list:
            pass
        else:
            raise ValueError("Unknown format for 'peaklabels'. "
                             "Must be boolean or list")
            return
        if vert_offset is None:
            vert_min, vert_max = axis.get_ylim()
            vert_offset = 0.05 * vert_max
        if horz_offset is None:
            horz_min, horz_max = axis.get_xlim()
            horz_offset = 0.01 * horz_max
        for i in range(0, len(peaklabels)):
            element, line = peaklabels[i].split('_')
            energy = (elements.elements[element]
                      ['Atomic_properties']
                      ['Xray_lines']
                      [line]
                      ['energy (keV)'])
            y_pos = spec.isig[energy].data + vert_offset
            x_pos = energy + horz_offset
            if y_pos > vert_max:
                y_pos = vert_max + 0.01 * vert_max
            if (x_pos < horz_min) or (x_pos > horz_max):
                pass
            else:
                axis.text(x=x_pos,
                          y=y_pos,
                          s=peaklabels[i],
                          rotation=90,
                          rotation_mode='anchor',
                          size=font_size)
    if out:
        return figure, axis
    else:
        return


def get_label_images(si, indices=None, plot=False, titles=None):
    """
    Segment results of SI decomposition.

    Args
    ----------
    si : Hyperspy Signal2D, EELSSpectrum, EDSSemSpectrum, or EDSTEMSpectrum
        SI datacube
    indices : list
        If provided, return segmented labels for only those components.
    plot : boolean
        If True, display a color figure showing the returned label images.
        Default is False.
    titles : list of strings
        If provided, label the individual images in the displayed figure.  If
        not provided, images are labeld as Component 0, Component 1, etc.

    Returns
    ----------
    labels : list of Hyperspy Signal2D instances

    """
    if not si.learning_results.decomposition_algorithm:
        raise ValueError('Decomposition has not been performed.')

    if not indices:
        indices = np.arange(0, si.learning_results.loadings.shape[1])

    labels = [None] * len(indices)
    masks = [None] * len(indices)
    for i in range(0, len(indices)):
        labels[i] = si.get_decomposition_loadings().inav[indices[i]]
        thresh = filters.threshold_otsu(labels[i].data)
        masks[i] = labels[i] > thresh
        labels[i] = hs.signals.Signal2D(measure.label(masks[i].data))
        if titles:
            labels[i].metadata.General.title = titles[i]
        else:
            labels[i].metadata.General.title = 'Component %s' % str(indices[i])

    if plot:
        max_vals = [1 + i.data.max() for i in labels]
        if not titles:
            titles = ['Component %s' % str(i)
                      for i in range(0, len(labels))]
        hs.plot.plot_images(labels, per_row=2, cmap='nipy_spectral',
                            vmax=max_vals, label=titles, axes_decor='off',
                            colorbar=None)
    return labels


def get_masked_intensity(si, label_im, line, bw):
    """
    Segment results of SI decomposition.

    Args
    ----------
    si : Hyperspy Signal2D, EELSSpectrum, EDSSemSpectrum, or EDSTEMSpectrum
        SI datacube.
    label_im : Hyperspy Signal2D
        Segmented image to use as mask. Must have the same navigation
        dimensions as si.
    line : str
        X-ray line for which to extract the masked intensity. Must be in the
        corrected format (e.g. 'Ni_Ka')
    bw : list of floats
        Pre-peak and post-peak windows to use for background fitting.

    Returns
    ----------
    intensites : NumPy array
        Integrated intensity for peak in the masked region

    """
    si.unfold()
    masked = si.deepcopy()
    intensities = [None] * label_im.data.max()
    for i in range(0, label_im.data.max()):
        mask = label_im == i + 1
        mask.unfold()
        masked.data = (mask.data * si.data.T).T
        result = masked.get_lines_intensity(xray_lines=[line, ],
                                            backround_windows=bw)[0]
        intensities[i] = result.sum().data[0]
    si.fold()
    intensities = np.array(intensities)
    return intensities


def get_volumes(label_im, scale):
    """
    Estimate volume of labeled region.

    Args
    ----------
    label_im : Hyperspy Signal2D
        Segmented image to use for volume estimation.
    scale : float
        Pixel size of the image.

    Returns
    ----------
    volumes : NumPy array
        Estimated volume for each segmented region in label_im

    """
    volumes = [None] * label_im.data.max()
    regions = measure.regionprops(label_im.data, coordinates='xy')
    for i in range(0, len(regions)):
        h = 1e-9 * scale * regions[i]['minor_axis_length']
        r = 1e-9 * scale * regions[i]['major_axis_length']
        volumes[i] = np.pi * r**2 * h
    volumes = np.array(volumes)
    return volumes


def get_tau_d(label_im, scale, tau):
    """
    Determine per dwell time per unit area.

    Args
    ----------
    label_im : Hyperspy Signal2D
        Segmented image to use for volume estimation.
    scale : float
        Pixel size of the image.
    tau : float or int
        Per pixel dwell time used for data collection in seconds.

    Returns
    ----------
    tau_d : NumPy array
        Dwell time per unit area of the segmented regions.

    """
    regions = measure.regionprops(label_im.data, coordinates='xy')
    tau_d = [None] * len(regions)
    for i in range(0, len(regions)):
        npix = regions[i]['area']
        tau_d[i] = tau / (npix * (scale * 1e-9)**2)
    tau_d = np.array(tau_d)
    return tau_d


def get_zeta_factor(si, label_im, line, bw=[4.0, 4.5, 11.8, 12.2],
                    i_probe=0.5e-9, tau=200e-3, rho=None):
    """
    Calculate the zeta factor for each segmented region.

    Args
    ----------
    si : Hyperspy Signal2D, EELSSpectrum, EDSSemSpectrum, or EDSTEMSpectrum
        SI datacube.
    label_im : Hyperspy Signal2D
        Segmented image to use as mask. Must have the same navigation
        dimensions as si.
    line : str
        X-ray line for which to extract the masked intensity. Must be in the
        corrected format (e.g. 'Ni_Ka')
    bw : list of floats
        Pre-peak and post-peak windows to use for background fitting.
    i_probe : float
        Probe current used for data collection in amps.
    tau : float
        Per pixel dwell time in seconds used for data collection.
    rho : float
        Density (g/cm^3) of the chosen element.

    Returns
    ----------
    counts_per_dose : NumPy array
        Estimated volume for each segmented region in label_im
    rho_v : NumPy array
    fit : NumPy array

    """
    Ne = 6.241e18               # Electrons per Coulomb

    """Extract the intensity of the chosen line in each masked reagion from
    the original SI"""
    counts = get_masked_intensity(si, label_im, line, bw)

    """Estimate the volume of each region assuming a cylindrical shape"""
    volumes = get_volumes(label_im, si.axes_manager[0].scale)

    """Calculate the dwell time per unit area for each segmented region"""
    tau_d = get_tau_d(label_im, si.axes_manager[0].scale, tau)

    """Calculate the counts per electron dose for each volume"""
    counts_per_dose = counts / (Ne * i_probe * tau_d)
    # counts_per_dose = np.append(0, counts_per_dose)

    """Calculate the amount of mass present in each volume"""
    rho_v = rho * volumes
    # rho_v = np.append(0, rho_v)

    """Perform a linear fit to rho_v vs. counts_per_dose"""
    zeta, b = np.polyfit(counts_per_dose, rho_v, 1)
    fit = zeta * counts_per_dose + b

    return counts_per_dose, rho_v, fit, zeta
