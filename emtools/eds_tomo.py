# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
EDS Tomography module for EMTools package

@author: Andrew Herzing
"""

import numpy as np
import hyperspy.api as hs
from skimage import measure, filters


def get_label_images(si_data, indices=None, plot=False, titles=None):
    """
    Segment results of SI decomposition.

    Args
    ----------
    si_data : Hyperspy Signal2D, EELSSpectrum, EDSSemSpectrum, or EDSTEMSpectrum
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
    if not si_data.learning_results.decomposition_algorithm:
        raise ValueError('Decomposition has not been performed.')

    if not indices:
        indices = np.arange(0, si_data.learning_results.loadings.shape[1])

    labels = [None] * len(indices)
    masks = [None] * len(indices)
    for i, _ in enumerate(indices):
        labels[i] = si_data.get_decomposition_loadings().inav[indices[i]]
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


def get_masked_intensity(si_data, label_im, line, windows):
    """
    Segment results of SI decomposition.

    Args
    ----------
    si_data : Hyperspy Signal2D, EELSSpectrum, EDSSemSpectrum, or EDSTEMSpectrum
        SI datacube.
    label_im : Hyperspy Signal2D
        Segmented image to use as mask. Must have the same navigation
        dimensions as si.
    line : str
        X-ray line for which to extract the masked intensity. Must be in the
        corrected format (e.g. 'Ni_Ka')
    windows : list of floats
        Pre-peak and post-peak windows to use for background fitting.

    Returns
    ----------
    intensites : NumPy array
        Integrated intensity for peak in the masked region

    """
    si_data.unfold()
    masked = si_data.deepcopy()
    intensities = [None] * label_im.data.max()
    for i in range(0, label_im.data.max()):
        mask = label_im == i + 1
        mask.unfold()
        masked.data = (mask.data * si_data.data.T).T
        result = masked.get_lines_intensity(xray_lines=[line, ],
                                            backround_windows=windows)[0]
        intensities[i] = result.sum().data[0]
    si_data.fold()
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
    for i, _ in enumerate(regions):
        height = 1e-9 * scale * regions[i]['minor_axis_length']
        radius = 1e-9 * scale * regions[i]['major_axis_length']
        volumes[i] = np.pi * radius**2 * height
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
    for i, _ in enumerate(regions):
        npix = regions[i]['area']
        tau_d[i] = tau / (npix * (scale * 1e-9)**2)
    tau_d = np.array(tau_d)
    return tau_d


def get_zeta_factor(si_data, label_im, line, windows=[4.0, 4.5, 11.8, 12.2],
                    i_probe=0.5e-9, tau=200e-3, rho=None):
    """
    Calculate the zeta factor for each segmented region.

    Args
    ----------
    si_data : Hyperspy Signal2D, EELSSpectrum, EDSSemSpectrum, or EDSTEMSpectrum
        SI datacube.
    label_im : Hyperspy Signal2D
        Segmented image to use as mask. Must have the same navigation
        dimensions as si.
    line : str
        X-ray line for which to extract the masked intensity. Must be in the
        corrected format (e.g. 'Ni_Ka')
    windows : list of floats
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
    e_per_coulumb = 6.242e18               # Electrons per Coulomb
    results = {}

    ### Extract the intensity of the chosen line in each masked reagion from
    ### the original SI
    counts = get_masked_intensity(si_data, label_im, line, windows)

    ### Estimate the volume of each region assuming a cylindrical shape
    volumes = get_volumes(label_im, si_data.axes_manager[0].scale)

    ### Calculate the dwell time per unit area for each segmented region
    tau_d = get_tau_d(label_im, si_data.axes_manager[0].scale, tau)

    ### Calculate the counts per electron dose for each volume
    counts_per_electron = counts / (e_per_coulumb * i_probe * tau_d)

    ### Calculate the amount of mass present in each volume
    rho_v = rho * volumes

    ### Perform a linear fit to rho_v vs. counts_per_dose
    zeta, slope = np.polyfit(np.append(0, counts_per_electron),
                             np.append(0, rho_v),
                             1)
    fit = zeta * counts_per_electron + slope

    results['counts'] = counts
    results['volumes'] = volumes
    results['tau_d'] = tau_d
    results['counts_per_electron'] = counts_per_electron
    return counts_per_electron, rho_v, fit, zeta
