# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
EDS module for EMTools package

@author: Andrew Herzing
"""

from matplotlib import pylab as plt
import hyperspy.api as hs
from skimage import measure, filters
import numpy as np
import pprint as pp
import imp


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
            energy = (hs.material.elements[element]
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
    Ne = 6.242e18               # Electrons per Coulomb
    results = {}

    """Extract the intensity of the chosen line in each masked reagion from
    the original SI"""
    counts = get_masked_intensity(si, label_im, line, bw)

    """Estimate the volume of each region assuming a cylindrical shape"""
    volumes = get_volumes(label_im, si.axes_manager[0].scale)

    """Calculate the dwell time per unit area for each segmented region"""
    tau_d = get_tau_d(label_im, si.axes_manager[0].scale, tau)

    """Calculate the counts per electron dose for each volume"""
    counts_per_electron = counts / (Ne * i_probe * tau_d)

    """Calculate the amount of mass present in each volume"""
    rho_v = rho * volumes

    """Perform a linear fit to rho_v vs. counts_per_dose"""
    zeta, b = np.polyfit(np.append(0, counts_per_electron),
                         np.append(0, rho_v),
                         1)
    fit = zeta * counts_per_electron + b

    results['counts'] = counts
    results['volumes'] = volumes
    results['tau_d'] = tau_d
    results['counts_per_electron'] = counts_per_electron
    return counts_per_electron, rho_v, fit, zeta


def calc_zeta_factor(s, element, line, thickness, ip=None, live_time=None,
                     bw=None, line_width=[5.0, 2.0]):
    Ne = 6.242e18
    if element not in s.metadata.Sample.elements:
        s.add_elements([element, ])
    if not bw:
        bw = s.estimate_background_windows(line_width=line_width)
    counts = s.get_lines_intensity([line, ],
                                   background_windows=bw)[0].data[0]

    rho = 1000 * hs.material.elements[element].Physical_properties.density_gcm3
    if not ip:
        if 'Acquisition_instrument.TEM.beam_current' in s.metadata:
            ip = 1e-9 * s.metadata.Acquisition_instrument.TEM.beam_current
        else:
            raise ValueError('Probe current not specified in metadata')
    if not live_time:
        if 'Acquisition_instrument.TEM.Detector.EDS.live_time' in s.metadata:
            live_time = s.metadata.Acquisition_instrument.TEM.\
                Detector.EDS.live_time
        else:
            raise ValueError('Live-time not specified in metadata')

    zeta = (rho * thickness * Ne * ip * live_time) / counts
    return zeta


def niox(spec, thickness=59, live_time=None, tilt=0, thickness_error=None,
         i_probe=None, display=True):
    """
    Calculate various detector characteristics from a nickel oxide spectrum.

    Args
    ----------
    data : Hyperspy Signal1D, EDSSemSpectrum, or EDSTEMSpectrum
        XEDS spectrum collected from NiOx film.
    live_time : float or int
        Spectrum acquisition time in seconds. Default is 200 seconds.
    thickness : float or int
        Thickness of NiOx film in nanometers. Default is 59 nm.
    tilt : float or int
        Specimen tilt in degrees.  Default is 0 degrees.
    thickness_error : float
        Error in thickness measurement given by manufacturer (+/- centimeters)
    current : float
        Probe current in amps. Default is 0.3 nA.
    display : bool
        If True, print the results to the terminal.

    Returns
    ----------
    results : Dict
        Dictionary containing all measured and calculated values.

    """
    # Check for experimental parameters in metadata
    if not live_time:
        if spec.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time > 0:
            live_time = \
                spec.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time
        else:
            raise ValueError('Spectrum acquisition time is not defined')

    if not thickness:
        raise ValueError('Specimen thickness not provided')

    if not i_probe:
        if spec.metadata.Acquisition_instrument.TEM.beam_current > 0:
            i_probe = spec.metadata.Acquisition_instrument.TEM.beam_current
        else:
            raise ValueError('Probe current is not defined')

    # Define parameters
    # rho : density of bulk NiOx (g/cm^3)
    # sigmaNi : ionization cross section for NiK (cm^2; 1 barn = 1e-24 cm^2)
    # Ne : electrons per Coulomb
    # w : fluoresence yield (unitless)
    # N : calculated number of Ni atoms per unit area; corrected for tilt

    rho = 6.67
    Ne = 6.242e18
    gmole_niox = 58.7 + 16.0
    N = 6.02e23 * rho / gmole_niox * thickness * 1e-7\
        / np.cos(tilt * np.pi / 180)
    sigmaNi = 255e-24
    dose = Ne * live_time * i_probe
    wNi = 0.414

    results = {}

    spec = spec.isig[2.:21.].deepcopy()
    spec.set_elements([])
    spec.set_lines([])
    spec.add_elements(['Co', 'Fe', 'Ni', 'O', 'Mo'])

    m = spec.create_model()

    m.fit()
    m.fit_background()
    # m.calibrate_energy_axis(calibrate='resolution', xray_lines=['Ni_Ka'])

    # m.calibrate_xray_lines('energy', ['Ni_Ka', 'Mo-Ka', 'Mo-La'], bound=10)
    # m.calibrate_xray_lines('width', ['Ni_Ka', 'Mo-Ka', 'Mo-La'], bound=10)

    '''Net Counts'''
    NiKa = m.components.Ni_Ka.A.value
    NiKb = m.components.Ni_Kb.A.value
    MoKa = m.components.Mo_Ka.A.value
    MoLa = m.components.Mo_La.A.value
    FeKa = m.components.Fe_Ka.A.value

    sigmaNiKa = np.sqrt(NiKa)
    sigmaNiKb = np.sqrt(NiKb)
    sigmaFeKa = np.sqrt(FeKa)
    sigmaMoKa = np.sqrt(MoKa)
    sigmaMoLa = np.sqrt(MoLa)

    results['Peaks'] = {}
    results['Peaks']['NiKa'] = {}
    results['Peaks']['NiKa']['Counts'] = NiKa
    results['Peaks']['NiKa']['Sigma'] = sigmaNiKa

    results['Peaks']['NiKb'] = {}
    results['Peaks']['NiKb']['Counts'] = NiKb
    results['Peaks']['NiKb']['Sigma'] = sigmaNiKb

    results['Peaks']['FeKa'] = {}
    results['Peaks']['FeKa']['Counts'] = FeKa
    results['Peaks']['FeKa']['Sigma'] = sigmaFeKa

    results['Peaks']['MoKa'] = {}
    results['Peaks']['MoKa']['Counts'] = MoKa
    results['Peaks']['MoKa']['Sigma'] = sigmaMoKa

    results['Peaks']['MoLa'] = {}
    results['Peaks']['MoLa']['Counts'] = MoLa
    results['Peaks']['MoLa']['Sigma'] = sigmaMoLa

    '''Energy Resolution'''
    results['Resolution'] = {}
    results['Resolution']['Height'] = NiKa
    results['Resolution']['Center'] = m.components.Ni_Ka.centre.value
    results['Resolution']['Sigma'] = m.components.Ni_Ka.sigma.value
    results['Resolution']['FWHM'] = {}
    results['Resolution']['FWHM']['NiKa'] = (1000.0 * m.components.Ni_Ka.fwhm)
    results['Resolution']['FWHM']['MnKa'] = \
        (0.926 * 1000.0 * m.components.Ni_Ka.fwhm)

    '''Peak to Background'''

    fwtm = 2 * np.sqrt(2 * np.log(10)) * m.components.Ni_Ka.sigma.value

    bckg1 = spec.isig[6.1 - 2 * fwtm:6.1].sum(0).data[0]
    bckg2 = spec.isig[8.7:8.7 + 2 * fwtm].sum(0).data[0]
    bckgavg = (bckg1 + bckg2) / 2
    bckgsingle = bckgavg / spec.isig[6.1 - 2 * fwtm:6.1].data.shape[0]

    totalpb = NiKa / bckgavg
    sigmaTotal = totalpb * np.sqrt((sigmaNiKa / NiKa)**2 +
                                   (np.sqrt(bckgavg) / bckgavg)**2)

    fiori = NiKa / bckgsingle
    sigmaFiori = fiori * np.sqrt((sigmaNiKa / NiKa)**2 +
                                 (np.sqrt(bckg1) / bckg1)**2 +
                                 (np.sqrt(bckg2) / bckg2)**2)

    results['FioriPB'] = {}
    results['FioriPB']['Value'] = fiori
    results['FioriPB']['Sigma'] = sigmaFiori
    results['TotalPB'] = {}
    results['TotalPB']['Value'] = totalpb
    results['TotalPB']['Sigma'] = sigmaTotal

    '''Hole Count'''
    holecountMo = NiKa / MoKa
    sigmaMo = holecountMo * np.sqrt((sigmaNiKa / NiKa)**2 +
                                    (sigmaMoKa / MoKa)**2)

    holecountFe = NiKa / FeKa
    sigmaFe = holecountFe * np.sqrt((sigmaNiKa / m.components.Ni_Ka.A.value)**2
                                    + (sigmaFeKa / FeKa)**2)

    results['HoleCount'] = {}
    results['HoleCount']['MoKa'] = {}
    results['HoleCount']['MoKa']['Value'] = holecountMo
    results['HoleCount']['MoKa']['Sigma'] = sigmaMo

    '''Mo K to L Ratio'''
    moklratio = MoKa / MoLa
    sigmaMokl = moklratio * np.sqrt((sigmaMoKa / MoKa)**2 +
                                    (sigmaMoLa / MoLa)**2)

    results['MoKL_Ratio'] = {}
    results['MoKL_Ratio']['Value'] = moklratio
    results['MoKL_Ratio']['Sigma'] = sigmaMokl

    '''Solid Angle and Efficiency'''
    omega = 4 * np.pi * (NiKa + NiKb) / (N * sigmaNi * wNi * dose)
    sigmaOmega = omega * np.sqrt((sigmaNiKa / NiKa)**2 +
                                 (sigmaNiKb / NiKb)**2 +
                                 (thickness_error / thickness)**2)

    efficiency = (NiKa + NiKb) / (live_time * i_probe * 1e9 * omega)
    sigmaEfficiency = efficiency * np.sqrt((sigmaNiKa / NiKa)**2 +
                                           (sigmaNiKb / NiKb)**2 +
                                           (sigmaOmega / omega)**2)
    results['Omega'] = {}
    results['Omega']['Value'] = omega
    results['Omega']['Sigma'] = sigmaOmega
    results['Efficiency'] = {}
    results['Efficiency']['Value'] = efficiency
    results['Efficiency']['Sigma'] = sigmaEfficiency

    '''Analysis Output'''

    if display:
        print('Results of NiOx Analysis')
        print('\n\tFilename:\t%s' % spec.metadata.General.original_filename)
        print('\tEnergy scale:\t%0.2f eV'
              % (1000 * spec.axes_manager[-1].scale))
        print('\n\tFilm thickness:\t\t%0.1f nm' % (thickness))
        print('\tAcquisition time:\t%0.1f s' % (live_time))
        print('\tProbe current:\t\t%0.2f nA' % (i_probe * 1e9))

        print('\n\tMeasured peak intensities')
        print('\tNet Ni-Ka peak height:\t\t%0.1f counts , sigma = %0.1f'
              % (NiKa, sigmaNiKa))
        print('\tNet Ni-Kb peak height:\t\t%0.1f counts , sigma = %0.1f'
              % (NiKb, sigmaNiKb))
        print('\tNet Fe-Ka peak height:\t\t%0.1f counts , sigma = %0.1f'
              % (FeKa, sigmaFeKa))
        print('\tNet Mo-Ka peak height:\t\t%0.1f counts , sigma = %0.1f'
              % (MoKa, sigmaMoKa))
        print('\tNet Mo-La peak height:\t\t%0.1f counts , sigma = %0.1f'
              % (MoLa, sigmaMoLa))

        print('\n******************** Energy Resolution ********************')
        print('\n\tFit results')
        print('\tNi-Ka peak height:\t%0.1f counts'
              % m.components.Ni_Ka.A.value)
        print('\tNi-Ka peak center:\t%0.3f keV'
              % m.components.Ni_Ka.centre.value)
        print('\tNi-Ka peak sigma:\t%0.1f eV'
              % (1000.0 * m.components.Ni_Ka.sigma.value))
        print('\n\tFWHM at Ni-Ka:\t\t%0.1f eV'
              % (1000.0 * m.components.Ni_Ka.fwhm))
        print('\n\tFWHM at Mn-Ka:\t\t%0.1f eV'
              % (0.926 * 1000.0 * m.components.Ni_Ka.fwhm))
        for i in m[1:]:
            if i.name not in ['Ni_Ka', 'Ni_Kb']:
                m.remove(i.name)

        m.plot(True)
        ax = plt.gca()
        ax.set_xlim([6., 10.])
        ax.legend(['Data', 'Model', 'Background', 'Ni_Ka', 'Ni_Kb'])

        print('\n******************** Peak to Background ********************')
        print('\n\tBackground (average):\t\t%0.1f counts' % bckgavg)
        print('\tBackground (single channel):\t%0.1f counts' % bckgsingle)

        print('\n\tFiori P/B:\t%0.1f' % fiori)
        print('\tError (95%%):\t%0.2f' % (2 * sigmaFiori))
        print('\tError (99%%):\t%0.2f' % (3 * sigmaFiori))

        print('\n\tTotal P/B:\t%0.1f' % totalpb)
        print('\tError (95%%):\t%0.2f' % (2 * sigmaTotal))
        print('\tError (99%%):\t%0.2f' % (3 * sigmaTotal))

        print('\n******************** Inverse hole-count ********************')
        print('\n\tInverse hole-count (Mo-Ka):\t%0.2f' % holecountMo)
        print('\tError (95%%):\t\t\t%0.2f' % (2 * sigmaMo))
        print('\tError (99%%):\t\t\t%0.2f' % (3 * sigmaMo))

        print('\n\tInverse hole-count (Fe-Ka):\t%0.2f' % holecountFe)
        print('\tError (95%%):\t\t\t%0.2f' % (2 * sigmaFe))
        print('\tError (99%%):\t\t\t%0.2f' % (3 * sigmaFe))

        print('\n******************** Mo K/L Ratio ********************')
        print('\n\tMo K/L ratio:\t%0.2f' % moklratio)
        print('\tError (95%%):\t%0.2f' % (2 * sigmaMokl))
        print('\tError (99%%):\t%0.2f' % (3 * sigmaMokl))

        print('\n******************** Solid-angle ********************')
        print('\tMeasured peak intensities')
        print('\n\tCollection angle:\t%0.4f sr' % omega)
        print('\tError (95%%):\t\t%0.4f sr' % (2 * sigmaOmega))
        print('\tError (99%%):\t\t%0.4f sr' % (3 * sigmaOmega))

        print('\n\tDetector efficiency:\t%0.3f cps/nA/sr' % efficiency)
        print('\tError (95%%):\t\t%0.3f cps/nA/sr' % (2 * sigmaEfficiency))
        print('\tError (99%%):\t\t%0.3f cps/nA/sr' % (3 * sigmaEfficiency))
        print('*****************************************************')
    return results


def get_counts_2063a(spec, method='model', energy_range=None, elements=None,
                     plot_results=False, verbose=False):
    """
    Extract peak intensities from spectrum collected from SRM-2063a

    composition = {'Mg': {'massfrac': 0.0797, 'uncertainty': 0.34},
                   'Si': {'massfrac': 0.2534, 'uncertainty': 0.98},
                   'Ca': {'massfrac': 0.1182, 'uncertainty': 0.37},
                   'Fe': {'massfrac': 0.1106, 'uncertainty': 0.88},
                   'O': {'massfrac': 0.432, 'uncertainty': 1.60},
                   'Ar': {'massfrac': 0.004, 'uncertainty': False}}

    Args
    ------
    spec : Hyperspy EDSSEMSpectrum or EDSTEMSpectrum
        Spectrum collected from SRM-2063a thin-film glass
    method : str
        If 'model', perform model-based peak intensity extraction. If
        'windows', use the three-window method.
    energy_range : list
        Low and high energy cutoff for fitting
    elements : list
        Elements to use for fitting.  If None, Mg, Si, Ca, Fe, O, and Ar will
        be used.
    plot_results : bool
        If True, plot the input spectrum along with the residuals.
    verbose : bool
        If True, print the results to the terminal
    """
    if not energy_range:
        temp = spec.deepcopy()
    else:
        temp = spec.isig[energy_range[0]:energy_range[1]].deepcopy()

    if method == 'model':
        # warnings.simplefilter(action='ignore', category=FutureWarning)
        # warnings.simplefilter(action='ignore', category=UserWarning)

        temp.set_elements([])
        if not elements:
            temp.add_elements(['C', 'Mg', 'Si', 'Ca', 'Fe', 'O', 'Ar', 'Cu', ])
        elif type(elements) is str:
            temp.add_elements([elements, ])
        elif type(elements) is list:
            temp.add_elements(elements)
        else:
            raise ValueError('Unknown type (%s) for elements.'
                             'Must be list or string.' %
                             type(elements))

        m = temp.create_model()
        m.remove(['Mg_Kb', 'Fe_Lb3', 'Fe_Ln', 'Ca_Ll', 'Ca_Ln', 'Ar_Kb'])
        m.free_xray_lines_width('all')
        m.free_sub_xray_lines_weight(['O_Ka', 'Si_Ka', 'Fe_Ka'])
        m.free_xray_lines_energy(['O_Ka', 'Fe_Ka', 'Si_Ka'])
        m['Fe_Kb'].sigma.bmin = 0.02
        m['Fe_Kb'].sigma.bmax = 1

        m.fit(bounded=True)
        m.fit_background()

        m.calibrate_energy_axis(calibrate='resolution')

        if verbose:
            print('Results for Peak Fit')
            print('**********************')
            result = m.get_lines_intensity(plot_result=True)
            print('Reduced Chi-Sq: %.2f\n' % m.red_chisq.data)
        else:
            result = m.get_lines_intensity(plot_result=False)

        output = {}
        for i in range(0, len(result)):
            line = result[i].metadata.Sample.xray_lines[0]
            if line in ['Ar_Ka', 'Ca_Ka', 'Fe_Ka', 'Mg_Ka', 'O_Ka', 'Si_Ka']:
                output[line] = {'counts': np.around(result[i].data[0], 2),
                                'uncertainty': np.nan}

        if plot_results:
            residuals = temp - m.as_signal()

            m.plot(True)
            ax = plt.gca()
            ax.plot(residuals)
            labels = ['Data', 'Model', 'Background']
            for i in m[1:]:
                labels.append(i.name)
            labels.append('Residual')
            ax.legend(labels)
            ax.set_ylim([-300, 1.1 * temp.data.max()])\

    elif method == 'windows':
        temp.set_elements([])
        temp.set_lines([])

        temp.add_lines(['Mg_Ka', 'Si_Ka', 'Ca_Ka', 'Fe_Ka', 'O_Ka', 'Ar_Ka'])
        ar_ka_bckg = [2.66, 2.76, 3.16, 3.26]
        ca_ka_bckg = [3.37, 3.47, 4.20, 4.31]
        fe_ka_bckg = [6.00, 6.13, 6.68, 6.81]
        mg_ka_bckg = [1.03, 1.10, 1.41, 1.48]
        o_ka_bckg = [0.34, 0.40, 0.79, 0.85]
        si_ka_bckg = [1.49, 1.57, 1.95, 2.03]
        bw = np.array([ar_ka_bckg,
                       ca_ka_bckg,
                       fe_ka_bckg,
                       mg_ka_bckg,
                       o_ka_bckg,
                       si_ka_bckg])

        if verbose:
            [ar_ka,
             ca_ka,
             fe_ka,
             mg_ka,
             o_ka,
             si_ka] = temp.get_lines_intensity(background_windows=bw,
                                               plot_result=True)
        else:
            [ar_ka,
             ca_ka,
             fe_ka,
             mg_ka,
             o_ka,
             si_ka] = temp.get_lines_intensity(background_windows=bw,
                                               plot_result=False)

        output = {'Ar_Ka': {'counts': ar_ka.data[0], 'uncertainty': np.nan},
                  'Ca_Ka': {'counts': ca_ka.data[0], 'uncertainty': np.nan},
                  'Fe_Ka': {'counts': fe_ka.data[0], 'uncertainty': np.nan},
                  'Mg_Ka': {'counts': mg_ka.data[0], 'uncertainty': np.nan},
                  'O_Ka': {'counts': o_ka.data[0], 'uncertainty': np.nan},
                  'Si_Ka': {'counts': si_ka.data[0], 'uncertainty': np.nan}}

    return output


def calc_zeta_factor_2063a(results, i_probe, live_time, tilt=0,
                           plot_result=False, verbose=False):
    """
    Calculate Zeta factor from a spectrum collected from 2063a SRM

    Args
    ------
    results : Dict
        Peak intensities extracted from 2063a spectrum
    i_probe : float
        Probe current in nA
    live_time : float or int
        Live time of spectrum collection
    tilt : float or int
        Specimen tilt in degrees
    plot_result : bool
        If True, plot calculated Zeta factors as a function of X-ray energy.
    verbose : bool
        If True, print the results to the terminal
    """
    composition = {'Mg': {'massfrac': 0.0797, 'uncertainty': 0.0034},
                   'Si': {'massfrac': 0.2534, 'uncertainty': 0.0098},
                   'Ca': {'massfrac': 0.1182, 'uncertainty': 0.0037},
                   'Fe': {'massfrac': 0.1106, 'uncertainty': 0.0088},
                   'O': {'massfrac': 0.432, 'uncertainty': 0.0160},
                   'Ar': {'massfrac': 0.004, 'uncertainty': False}}

    rho = 3100
    rho_sigma = 300
    thickness = 76e-9 / np.cos(tilt * np.pi / 180)
    thickness_sigma = 4e-9
    dose = i_probe * live_time * 6.242e18

    zeta_mg = rho * thickness * composition['Mg']['massfrac'] * \
        dose / results['Mg_Ka']['counts']
    zeta_mg_sigma = np.sqrt((composition['Mg']['uncertainty'] /
                             composition['Mg']['massfrac'])**2 +
                            (2 * np.sqrt(results['Mg_Ka']['counts']) /
                             results['Mg_Ka']['counts'])**2 +
                            (thickness_sigma / thickness)**2 +
                            (rho_sigma / rho)**2) * zeta_mg

    zeta_si = rho * thickness * composition['Si']['massfrac'] * \
        dose / results['Si_Ka']['counts']
    zeta_si_sigma = np.sqrt((composition['Si']['uncertainty'] /
                             composition['Si']['massfrac'])**2 +
                            (2 * np.sqrt(results['Si_Ka']['counts']) /
                             results['Si_Ka']['counts'])**2 +
                            (thickness_sigma / thickness)**2 +
                            (rho_sigma / rho)**2) * zeta_si

    zeta_ca = rho * thickness * composition['Ca']['massfrac'] * \
        dose / results['Ca_Ka']['counts']
    zeta_ca_sigma = np.sqrt((composition['Ca']['uncertainty'] /
                             composition['Ca']['massfrac'])**2 +
                            (2 * np.sqrt(results['Ca_Ka']['counts']) /
                             results['Ca_Ka']['counts'])**2 +
                            (thickness_sigma / thickness)**2 +
                            (rho_sigma / rho)**2) * zeta_ca

    zeta_fe = rho * thickness * composition['Fe']['massfrac'] * \
        dose / results['Fe_Ka']['counts']
    zeta_fe_sigma = np.sqrt((composition['Fe']['uncertainty'] /
                             composition['Fe']['massfrac'])**2 +
                            (2 * np.sqrt(results['Fe_Ka']['counts']) /
                             results['Fe_Ka']['counts'])**2 +
                            (thickness_sigma / thickness)**2 +
                            (rho_sigma / rho)**2) * zeta_fe

    zeta_o = rho * thickness * composition['O']['massfrac'] * \
        dose / results['O_Ka']['counts']
    zeta_o_sigma = np.sqrt((composition['O']['uncertainty'] /
                            composition['O']['massfrac'])**2 +
                           (2 * np.sqrt(results['O_Ka']['counts']) /
                            results['O_Ka']['counts'])**2 +
                           (thickness_sigma / thickness)**2 +
                           (rho_sigma / rho)**2) * zeta_o

    if plot_result:
        xray_energies = [hs.material.elements[i].Atomic_properties.
                         Xray_lines['Ka']['energy_keV'] for i in
                         ['Mg', 'Si', 'Ca', 'Fe', 'O']]
        plt.figure()
        plt.scatter(xray_energies, [zeta_mg, zeta_si, zeta_ca,
                                    zeta_fe, zeta_o])

    zeta_factors = {'Mg_Ka': {'zeta_factor': np.round(zeta_mg, 2),
                              'zeta_factor_sigma': np.round(zeta_mg_sigma, 2)},
                    'Si_Ka': {'zeta_factor': np.round(zeta_si, 2),
                              'zeta_factor_sigma': np.round(zeta_si_sigma, 2)},
                    'Ca_Ka': {'zeta_factor': np.round(zeta_ca, 2),
                              'zeta_factor_sigma': np.round(zeta_ca_sigma, 2)},
                    'Fe_Ka': {'zeta_factor': np.round(zeta_fe, 2),
                              'zeta_factor_sigma': np.round(zeta_fe_sigma, 2)},
                    'O_Ka': {'zeta_factor': np.round(zeta_o, 2),
                             'zeta_factor_sigma': np.round(zeta_o_sigma, 2)}
                    }
    if verbose:
        pp.pprint(zeta_factors)
    return zeta_factors


def simulate_eds_spectrum(elements, ka_amplitude=None, nchannels=2048,
                          energy_resolution=135, energy_per_channel=0.01,
                          background=False, noise=False, beam_energy=300):
    """
    Simulate a simple XEDS spectrum containing K-lines

    Args
    ------
    elements : list
        Elements to include in the simulated spectrum
    ka_amplitude : float or int
        Amplitude of the Gaussian K-alpha peak to simulate
    nchannels : int
        Number of channels in the simulated spectrum
    energy_resolution : float
        Energy resolution of the simulated spectrum in eV
    energy_per_channel : float
        Energy per channel in the simulated spectrum in keV
    background : bool
        If True, include background in the simulated spectrum
    noise : bool
        If True, include Poissonian noise in the simulated spectrum
    beam_energy : float
        Beam energy in keV to use in the simulated spectrum.
    """

    if not ka_amplitude:
        ka_amplitude = 1000 * np.ones(len(elements))

    s = hs.signals.EDSTEMSpectrum(np.ones(nchannels))
    s.axes_manager[0].scale = energy_per_channel
    s.axes_manager[0].units = 'keV'
    s.axes_manager[0].offset = 0
    s.set_microscope_parameters(beam_energy=300)
    #                             energy_resolution_MnKa=120)
    s.metadata.General.original_filename = \
        ('%s EDS Simluation.msa' % str(elements))
    s.add_elements(elements)
    x_axis = s.axes_manager[0].axis

    count = 0
    for k in elements:
        lines = (hs.material.elements[k]
                                     ['Atomic_properties']
                                     ['Xray_lines'].keys())
        for i in lines:
            energy = (hs.material.elements[k]
                                          ['Atomic_properties']
                                          ['Xray_lines']
                                          [i]
                                          ['energy_keV'])
            weight = (hs.material.elements[k]
                                          ['Atomic_properties']
                                          ['Xray_lines']
                                          [i]
                                          ['weight'])
            A = weight * ka_amplitude[count]
            sigma = 0.001 * energy_resolution / (2 * np.sqrt(2 * np.log(2)))

            peak = (A / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-(x_axis - energy)**2
                    / (2 * sigma**2)))

            s.data += peak
        count += 1
    return s


def calc_solid_angle(material, xray_line, counts, thickness,
                     beam_energy, probe_current, live_time, tilt=0,
                     verbose=False):
    """
    Calcualte the solid angle of a detectr from measured spectrum

    Args
    ------
    material : str
        Material from which the spectrum was collected.  Must be either 'NiOx'
        or '2063a'
    xray_line : str
        X-ray line to use for the calculation
    counts : int
        Number of X-ray counts detected for the chosen line
    thickness : float
        Thickness of the specimen in nanometers analyzed
    beam_energy : float
        Energy of the electrons in keV used to collect the spectrum
    probe_current : float
        Probe current in nA used to generat the spectrum
    live_time : float
        Spectrum acquisition live time in seconds
    tilt : float
        Specimen tilt in degrees
    vebose : bool
        If True, print parameters and calculated results to the terminal
    """

    element, line = xray_line.split('_')
    xray_energy = hs.material.elements[element]\
                             .Atomic_properties\
                             .Xray_lines[line]\
                             .energy_keV
    eff_thickness = thickness / np.cos(np.pi * tilt / 180)
    mat = Material(material, beam_energy)
    w = mat.xray_lines[xray_line]['w']
    sigma = mat.xray_lines[xray_line]['sigma'] * 1e4
    N_atoms = mat.get_atoms_per_volume(element) *\
        (eff_thickness * 1e-7)
    Ne = probe_current * 1e-9 * live_time / 1.6e-19
    nu = counts / (N_atoms * sigma * w * Ne)
    omega = 4 * np.pi * nu
    omega = np.round(omega, 3)

    if verbose:
        print('Detector Solid-angle Calculation')
        print('*************************************')
        print('Beam energy (keV): %s' % str(beam_energy))
        print('Probe current (nA): %s' % str(probe_current))
        print('Live time (s): %s' % str(live_time))
        print('Electron dose: %.2e' % Ne)
        print('Effective sample thickness (nm): %.1f\n' % eff_thickness)
        print('X-ray line: %s @ %.2f keV' % (xray_line, xray_energy))
        print('Counts detected: %s' % str(counts))
        print('Ionization Cross-section (cm^2): %.2e' % sigma)
        print('Fluorescence Yield: %.3f' % w)
        print('Atoms per Unit Area (cm^-2): %.2e\n' % N_atoms)
        print('Collection Efficiency: %.2f %%' % (100 * nu))
        print('Collection Solid-angle (srs): %.3f' % omega)

    return omega


class Material:

    def __init__(self, name, beam_energy=300, thickness=None,
                 thickness_sigma=None):
        mats = ['NiOx', '2063a']

        if name in mats:
            pass
        else:
            raise ValueError("Unknown material %s. "
                             "Must be one of the following: "
                             "%s" % (name, ', '.join(mats)))
        self.name = name
        self.beam_energy = beam_energy
        self.xray_lines = None
        self.elements = None
        self.density = None
        self.density_sigma = None
        self.composition_by_atom = None
        self.composition_by_mass = None
        self.molar_mass = None
        self.total_atoms_per_gram = None

        if name == 'NiOx':
            self.elements = ['Ni', 'O']
            self.xray_lines = {'Ni_Ka': {'w': np.nan, 'sigma': np.nan},
                               'O_Ka': {'w': np.nan, 'sigma': np.nan}}
            self.get_xray_line_properties()
            self.density = 6.67
            self.density_sigma = np.nan
            if not thickness:
                self.thickness = 59
            if not thickness_sigma:
                self.thickness_sigma = 4
            self.composition_by_atom = {'Ni': {'atom_fraction': 0.5,
                                               'sigma': np.nan},
                                        'O': {'atom_fraction': 0.5,
                                              'sigma': np.nan}}
            self.molar_mass = self.get_molar_mass()
            self.composition_by_mass = self.at_to_wt()
            self.total_atoms_per_gram = self.get_atoms_per_gram()

        elif name == '2063a':
            self.elements = ['Mg', 'Si', 'Ca', 'Fe', 'O', 'Ar']
            self.xray_lines = {'Mg_Ka': {'w': np.nan, 'sigma': np.nan},
                               'Si_Ka': {'w': np.nan, 'sigma': np.nan},
                               'Ca_Ka': {'w': np.nan, 'sigma': np.nan},
                               'Fe_Ka': {'w': np.nan, 'sigma': np.nan},
                               'O_Ka': {'w': np.nan, 'sigma': np.nan},
                               'Ar_Ka': {'w': np.nan, 'sigma': np.nan}}
            self.get_xray_line_properties()
            self.density = 3.1
            self.density_sigma = 0.3
            if not thickness:
                self.thickness = 76
            if not thickness_sigma:
                self.thickness_sigma = 4
            self.composition_by_mass = {'Mg': {'mass_fraction': 0.0797,
                                               'sigma': 0.0034},
                                        'Si': {'mass_fraction': 0.2534,
                                               'sigma': 0.0098},
                                        'Ca': {'mass_fraction': 0.1182,
                                               'sigma': 0.0037},
                                        'Fe': {'mass_fraction': 0.1106,
                                               'sigma': 0.0088},
                                        'O': {'mass_fraction': 0.432,
                                              'sigma': 0.016},
                                        'Ar': {'mass_fraction': 0.004,
                                               'sigma': np.nan}}
            self.total_atoms_per_gram = self.get_atoms_per_gram()
            self.composition_by_atom = self.wt_to_at()
            self.molar_mass = self.get_molar_mass()

    def get_xray_line_properties(self):
        if not self.xray_lines:
            raise ValueError('No X-ray lines defined!')
        datapath = imp.find_module("emtools")[1] + "/data/"
        w = np.loadtxt(datapath + 'FluorescenceYield.txt')
        sigma = np.loadtxt(datapath +
                           "AbsoluteIonizationCrossSection" +
                           "BoteSalvat2008_KShell_%skeV.txt" %
                           str(self.beam_energy))
        for i in self.xray_lines:
            element = i.split('_')[0]
            Z = hs.material.elements[element].General_properties.Z
            idx = np.where(w[:, 0] == Z)[0][0]
            self.xray_lines[i]['w'] = w[idx, 1]
            self.xray_lines[i]['sigma'] = sigma[idx, 1]
        return

    def get_atoms_per_gram(self):
        total_atoms_per_gram = 0
        for i in self.composition_by_mass:
            total_atoms_per_gram +=\
                self.composition_by_mass[i]['mass_fraction']\
                / hs.material.elements[i].General_properties\
                                         .atomic_weight
        total_atoms_per_gram *= 6.02e23
        return total_atoms_per_gram

    def get_molar_mass(self):
        molar_mass = 0
        for i in self.composition_by_atom:
            molar_mass +=\
                100 * self.composition_by_atom[i]['atom_fraction']\
                * hs.material.elements[i].General_properties\
                                         .atomic_weight
        return molar_mass

    def wt_to_at(self):
        composition_by_atom = {}
        for i in self.composition_by_mass:
            atoms = (self.composition_by_mass[i]['mass_fraction']
                     / hs.material.elements[i]
                                  .General_properties
                                  .atomic_weight
                     * 6.02e23)
            atom_fraction = atoms / self.total_atoms_per_gram
            composition_by_atom[i] = {'atom_fraction': atom_fraction}
        return composition_by_atom

    def at_to_wt(self):
        composition_by_mass = {}
        for i in self.composition_by_atom:
            mass = (100 * self.composition_by_atom[i]['atom_fraction']
                    * hs.material.elements[i]
                                 .General_properties
                                 .atomic_weight)
            mass_fraction = mass / self.molar_mass
            composition_by_mass[i] = {'mass_fraction': mass_fraction}
        return composition_by_mass

    def get_atoms_per_volume(self, element):
        mass_frac = self.composition_by_mass[element]['mass_fraction']
        atomic_weight = hs.material.elements[element]\
            .General_properties\
            .atomic_weight
        atoms_per_volume = (6.02e23
                            * mass_frac / atomic_weight
                            * self.density)
        return atoms_per_volume
