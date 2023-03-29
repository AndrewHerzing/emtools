# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
EDS module for EMTools package.

@author: Andrew Herzing
"""

import imp
from matplotlib import pylab as plt
import hyperspy.api as hs
import numpy as np

datapath = imp.find_module("emtools")[1] + "/data/"


def get_test_spectrum(material='2063a'):
    """
    Load a reference spectrum from a chosen test material.

    Args
    -----
    material : str
        Name of material. Current options are '2063a' and 'NiOx'.

    Returns
    -----
    spec : Hyperspy Signal1D
        Spectrum from chosen material
    """
    allowed_mats = ['2063a', 'niox']

    if material.lower() not in allowed_mats:
        raise ValueError('Unknown material. Must be one of %s.' %
                         ', '.join(allowed_mats))

    spec = hs.load(datapath + 'XEDS_' + material + '.hdf5')
    return spec


def get_detector_efficiency(detector_name):
    """
    Read detecor efficiency from database.

    Args
    -----
    detector_name : str
        Name of the detector

    Returns
    -----
    detector_efficiency : Hyperspy Signal1D
        Detector efficiency curve as calculated by NIST DTSA-II
    """
    detectors = ['OctaneT', ]
    if detector_name in detectors:
        pass
    else:
        raise ValueError("Unknown detector %s. "
                         "Must be one of the following: "
                         "%s" % (detector_name, ', '.join(detectors)))

    detector_efficiency = np.loadtxt(datapath + detector_name + '_DetectorEfficiencyCurve.txt')
    detector_efficiency = hs.signals.Signal1D(detector_efficiency)
    detector_efficiency.axes_manager[0].name = 'Energy'
    detector_efficiency.axes_manager[0].units = 'keV'
    detector_efficiency.axes_manager[0].scale = 0.01
    return detector_efficiency


def calc_zeta_factor(spec, element, line, thickness, i_probe=None, live_time=None, windows=None, line_width=[5.0, 2.0]):
    """
    Calculate Zeta factor from a spectrum.

    Args
    -----
    spec : Hyperspy EDSSpectrum
        Spectrum used to calculate Zeta factor
    element : string
        Element for which to calculate Zeta factor
    line : string
        X-ray line to calculate Zeta factor
    thickness : float
        Specimen thickness
    i_probe : float
        Probe current
    live_time : float
        Spectral integration time
    windows : list of floats
        Windows for background subtraction
    line_width : list of floats
        Line width

    Returns
    -----
    zeta : float
        Calculated Zeta factor
    """
    electrons_per_coulomb = 6.242e18
    if element not in spec.metadata.Sample.elements:
        spec.add_elements([element, ])
    if not windows:
        windows = spec.estimate_background_windows(line_width=line_width)
    counts = spec.get_lines_intensity([line, ],
                                      background_windows=windows)[0].data[0]

    rho = 1000 * hs.material.elements[element].Physical_properties.density_gcm3
    if not i_probe:
        if 'Acquisition_instrument.TEM.beam_current' in spec.metadata:
            i_probe = 1e-9 *\
                spec.metadata.Acquisition_instrument.TEM.beam_current
        else:
            raise ValueError('Probe current not specified in metadata')
    if not live_time:
        key_check = 'Acquisition_instrument.TEM.Detector.EDS.live_time'
        if key_check in spec.metadata:
            live_time = spec.metadata.Acquisition_instrument.TEM.\
                Detector.EDS.live_time
        else:
            raise ValueError('Live-time not specified in metadata')

    zeta = (rho * thickness * electrons_per_coulomb * i_probe * live_time) / counts
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
    electrons_per_coulomb = 6.242e18
    gmole_niox = 58.7 + 16.0
    n_atoms = 6.02e23 * rho / gmole_niox * thickness * 1e-7\
        / np.cos(tilt * np.pi / 180)
    sigma_ni = 255e-24
    dose = electrons_per_coulomb * live_time * i_probe
    w_ni = 0.414

    results = {}

    spec = spec.isig[2.:21.].deepcopy()
    spec.set_elements([])
    spec.set_lines([])
    spec.add_elements(['Co', 'Fe', 'Ni', 'O', 'Mo'])

    model = spec.create_model()

    model.fit()
    model.fit_background()
    # model.calibrate_energy_axis(calibrate='resolution', xray_lines=['Ni_Ka'])

    # model.calibrate_xray_lines('energy', ['Ni_Ka', 'Mo-Ka', 'Mo-La'],
    #                            bound=10)
    # model.calibrate_xray_lines('width', ['Ni_Ka', 'Mo-Ka', 'Mo-La'],
    #                            bound=10)

    # Net Counts
    ni_ka = model.components.Ni_Ka.A.value
    ni_kb = model.components.Ni_Kb.A.value
    mo_ka = model.components.Mo_Ka.A.value
    mo_la = model.components.Mo_La.A.value
    fe_ka = model.components.Fe_Ka.A.value

    sigma_ni_ka = np.sqrt(ni_ka)
    sigma_ni_kb = np.sqrt(ni_kb)
    sigma_fe_ka = np.sqrt(fe_ka)
    sigma_mo_ka = np.sqrt(mo_ka)
    sigma_mo_la = np.sqrt(mo_la)

    results['Peaks'] = {}
    results['Peaks']['NiKa'] = {}
    results['Peaks']['NiKa']['Counts'] = ni_ka
    results['Peaks']['NiKa']['Sigma'] = sigma_ni_ka

    results['Peaks']['NiKb'] = {}
    results['Peaks']['NiKb']['Counts'] = ni_kb
    results['Peaks']['NiKb']['Sigma'] = sigma_ni_kb

    results['Peaks']['FeKa'] = {}
    results['Peaks']['FeKa']['Counts'] = fe_ka
    results['Peaks']['FeKa']['Sigma'] = sigma_fe_ka

    results['Peaks']['MoKa'] = {}
    results['Peaks']['MoKa']['Counts'] = mo_ka
    results['Peaks']['MoKa']['Sigma'] = sigma_mo_ka

    results['Peaks']['MoLa'] = {}
    results['Peaks']['MoLa']['Counts'] = mo_la
    results['Peaks']['MoLa']['Sigma'] = sigma_mo_la

    # Energy Resolution
    results['Resolution'] = {}
    results['Resolution']['Height'] = ni_ka
    results['Resolution']['Center'] = model.components.Ni_Ka.centre.value
    results['Resolution']['Sigma'] = model.components.Ni_Ka.sigma.value
    results['Resolution']['FWHM'] = {}
    results['Resolution']['FWHM']['NiKa'] =\
        (1000.0 * model.components.Ni_Ka.fwhm)
    results['Resolution']['FWHM']['MnKa'] = \
        (0.926 * 1000.0 * model.components.Ni_Ka.fwhm)

    # Peak to Background

    fwtm = 2 * np.sqrt(2 * np.log(10)) * model.components.Ni_Ka.sigma.value

    bckg1 = spec.isig[6.1 - 2 * fwtm:6.1].sum(0).data[0]
    bckg2 = spec.isig[8.7:8.7 + 2 * fwtm].sum(0).data[0]
    bckgavg = (bckg1 + bckg2) / 2
    bckgsingle = bckgavg / spec.isig[6.1 - 2 * fwtm:6.1].data.shape[0]

    totalpb = ni_ka / bckgavg
    sigma_total = totalpb * np.sqrt((sigma_ni_ka / ni_ka)**2 + (np.sqrt(bckgavg) / bckgavg)**2)

    fiori = ni_ka / bckgsingle
    sigma_fiori = fiori * np.sqrt((sigma_ni_ka / ni_ka)**2 + (np.sqrt(bckg1) / bckg1)**2 + (np.sqrt(bckg2) / bckg2)**2)

    results['FioriPB'] = {}
    results['FioriPB']['Value'] = fiori
    results['FioriPB']['Sigma'] = sigma_fiori
    results['TotalPB'] = {}
    results['TotalPB']['Value'] = totalpb
    results['TotalPB']['Sigma'] = sigma_total

    # Hole Count
    holecount_mo = ni_ka / mo_ka
    sigma_mo = holecount_mo * np.sqrt((sigma_ni_ka / ni_ka)**2 + (sigma_mo_ka / mo_ka)**2)

    holecount_fe = ni_ka / fe_ka
    sigma_fe = holecount_fe *\
        np.sqrt((sigma_ni_ka / model.components.Ni_Ka.A.value)**2 + (sigma_fe_ka / fe_ka)**2)

    results['HoleCount'] = {}
    results['HoleCount']['MoKa'] = {}
    results['HoleCount']['MoKa']['Value'] = holecount_mo
    results['HoleCount']['MoKa']['Sigma'] = sigma_mo

    # Mo K to L Ratio
    mo_kl_ratio = mo_ka / mo_la
    sigma_mo_kl = mo_kl_ratio * np.sqrt((sigma_mo_ka / mo_ka)**2 + (sigma_mo_la / mo_la)**2)

    results['MoKL_Ratio'] = {}
    results['MoKL_Ratio']['Value'] = mo_kl_ratio
    results['MoKL_Ratio']['Sigma'] = sigma_mo_kl

    # Solid Angle and Efficiency
    omega = 4 * np.pi * (ni_ka + ni_kb) / (n_atoms * sigma_ni * w_ni * dose)
    sigma_omega = omega * np.sqrt((sigma_ni_ka / ni_ka)**2 + (sigma_ni_kb / ni_kb)**2 + (thickness_error / thickness)**2)

    efficiency = (ni_ka + ni_kb) / (live_time * i_probe * 1e9 * omega)
    sigma_efficiency = efficiency * np.sqrt((sigma_ni_ka / ni_ka)**2 + (sigma_ni_kb / ni_kb)**2 + (sigma_omega / omega)**2)
    results['Omega'] = {}
    results['Omega']['Value'] = omega
    results['Omega']['Sigma'] = sigma_omega
    results['Efficiency'] = {}
    results['Efficiency']['Value'] = efficiency
    results['Efficiency']['Sigma'] = sigma_efficiency

    # Analysis Output

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
              % (ni_ka, sigma_ni_ka))
        print('\tNet Ni-Kb peak height:\t\t%0.1f counts , sigma = %0.1f'
              % (ni_kb, sigma_ni_kb))
        print('\tNet Fe-Ka peak height:\t\t%0.1f counts , sigma = %0.1f'
              % (fe_ka, sigma_fe_ka))
        print('\tNet Mo-Ka peak height:\t\t%0.1f counts , sigma = %0.1f'
              % (mo_ka, sigma_mo_ka))
        print('\tNet Mo-La peak height:\t\t%0.1f counts , sigma = %0.1f'
              % (mo_la, sigma_mo_la))

        print('\n******************** Energy Resolution ********************')
        print('\n\tFit results')
        print('\tNi-Ka peak height:\t%0.1f counts'
              % model.components.Ni_Ka.A.value)
        print('\tNi-Ka peak center:\t%0.3f keV'
              % model.components.Ni_Ka.centre.value)
        print('\tNi-Ka peak sigma:\t%0.1f eV'
              % (1000.0 * model.components.Ni_Ka.sigma.value))
        print('\n\tFWHM at Ni-Ka:\t\t%0.1f eV'
              % (1000.0 * model.components.Ni_Ka.fwhm))
        print('\n\tFWHM at Mn-Ka:\t\t%0.1f eV'
              % (0.926 * 1000.0 * model.components.Ni_Ka.fwhm))
        for i in model[1:]:
            if i.name not in ['Ni_Ka', 'Ni_Kb']:
                model.remove(i.name)

        model.plot(True)
        axis = plt.gca()
        axis.set_xlim([6., 10.])
        axis.legend(['Data', 'Model', 'Background', 'Ni_Ka', 'Ni_Kb'])

        print('\n******************** Peak to Background ********************')
        print('\n\tBackground (average):\t\t%0.1f counts' % bckgavg)
        print('\tBackground (single channel):\t%0.1f counts' % bckgsingle)

        print('\n\tFiori P/B:\t%0.1f' % fiori)
        print('\tError (95%%):\t%0.2f' % (2 * sigma_fiori))
        print('\tError (99%%):\t%0.2f' % (3 * sigma_fiori))

        print('\n\tTotal P/B:\t%0.1f' % totalpb)
        print('\tError (95%%):\t%0.2f' % (2 * sigma_total))
        print('\tError (99%%):\t%0.2f' % (3 * sigma_total))

        print('\n******************** Inverse hole-count ********************')
        print('\n\tInverse hole-count (Mo-Ka):\t%0.2f' % holecount_mo)
        print('\tError (95%%):\t\t\t%0.2f' % (2 * sigma_mo))
        print('\tError (99%%):\t\t\t%0.2f' % (3 * sigma_mo))

        print('\n\tInverse hole-count (Fe-Ka):\t%0.2f' % holecount_fe)
        print('\tError (95%%):\t\t\t%0.2f' % (2 * sigma_fe))
        print('\tError (99%%):\t\t\t%0.2f' % (3 * sigma_fe))

        print('\n******************** Mo K/L Ratio ********************')
        print('\n\tMo K/L ratio:\t%0.2f' % mo_kl_ratio)
        print('\tError (95%%):\t%0.2f' % (2 * sigma_mo_kl))
        print('\tError (99%%):\t%0.2f' % (3 * sigma_mo_kl))

        print('\n******************** Solid-angle ********************')
        print('\tMeasured peak intensities')
        print('\n\tCollection angle:\t%0.4f sr' % omega)
        print('\tError (95%%):\t\t%0.4f sr' % (2 * sigma_omega))
        print('\tError (99%%):\t\t%0.4f sr' % (3 * sigma_omega))

        print('\n\tDetector efficiency:\t%0.3f cps/nA/sr' % efficiency)
        print('\tError (95%%):\t\t%0.3f cps/nA/sr' % (2 * sigma_efficiency))
        print('\tError (99%%):\t\t%0.3f cps/nA/sr' % (3 * sigma_efficiency))
        print('*****************************************************')
    return results


def simulate_eds_spectrum(elements, ka_amplitude=None, nchannels=2048,
                          energy_resolution=135, energy_per_channel=0.01,
                          beam_energy=300):
    """
    Simulate a simple XEDS spectrum containing K-lines.

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
    beam_energy : float
        Beam energy in keV to use in the simulated spectrum.
    """
    if not ka_amplitude:
        ka_amplitude = 1000 * np.ones(len(elements))

    spec = hs.signals.EDSTEMSpectrum(np.ones(nchannels))
    spec.axes_manager[0].scale = energy_per_channel
    spec.axes_manager[0].units = 'keV'
    spec.axes_manager[0].offset = 0
    spec.set_microscope_parameters(beam_energy=beam_energy)
    spec.metadata.General.original_filename = \
        ('%s EDS Simluation.msa' % str(elements))
    spec.add_elements(elements)
    x_axis = spec.axes_manager[0].axis

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
            amplitude = weight * ka_amplitude[count]
            sigma = 0.001 * energy_resolution / (2 * np.sqrt(2 * np.log(2)))

            peak = (amplitude / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x_axis - energy)**2 / (2 * sigma**2)))

            spec.data += peak
        count += 1
    return spec


class QuantSpec:
    """
    Class to create materials with provided composition.

    Attributes
    -----
    name : str
        Identity of the material.  Acceptable materials are 'NiOx' and '2063a'
    beam_energy : float
        Beam enegy in keV associated with analysis of the material
    xray_lines : dict
        X-ray lines to includ in analysis.  Includes the ionization cross-
        section and fluorescence yield for each line.
    elements : list
        Elements to include in the analysis
    density : float
        Density in g/cm^3 of the material
    density_sigma : float
        Uncertainty in the material density in g/cm^3
    composition_by_atom : dict
        Composition of the material by atom. Contains atom fractions and
        uncertainty for all elements.
    composition_by_mass : dict
        Composition of the material by mass. Contains mass fractions and
        uncertainty for all elements.
    molar_mass : float
        Molar mass of the material in grams
    total_atoms_per_gram : float
        Number density of atoms per gram of the material
    zeta_factors : list
        Zeta factor for each element

    """

    def __init__(self, spec, material, beam_energy=None, thickness=None,
                 thickness_sigma=None, live_time=None, probe_current=None,
                 specimen_tilt=None):
        """
        Construct an instance of the Material class.

        Args
        -----
        material : str
            Identity of the material.  Acceptable materials are 'NiOx' and
            '2063a'
        beam_energy : float
            Beam enegy in keV associated with analysis of the material. Default
            is 300 keV
        thickness : float
            Thickness of the specimen in nanomaters
        thickness_sigma : float
            Uncertainty in specimen thickness in nanometers
        """
        known_materials = ['NiOx', '2063a']

        if material in known_materials:
            pass
        else:
            raise ValueError("Unknown material %s. "
                             "Must be one of the following: "
                             "%s" % (material, ', '.join(known_materials)))
        self.material = material
        self.xray_lines = None
        self.elements = None
        self.density = None
        self.density_sigma = None
        self.composition_by_atom = None
        self.composition_by_mass = None
        self.molar_mass = None
        self.total_atoms_per_gram = None
        self.spec = spec
        self.thickness = thickness
        self.thickness_sigma = thickness_sigma
        self.intensities = None
        self.zeta_factors = None

        if specimen_tilt:
            self.specimen_tilt = specimen_tilt
        elif spec.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha:
            self.specimen_tilt = \
                spec.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha
        else:
            raise ValueError('Specimen tilt is not defined')
        if live_time:
            self.live_time = live_time
        elif spec.metadata.Acquisition_instrument\
                          .TEM.Detector.EDS.live_time > 0:
            self.live_time = \
                spec.metadata.Acquisition_instrument\
                             .TEM.Detector.EDS.live_time
        else:
            raise ValueError('Spectrum acquisition time is not defined')

        if probe_current:
            self.probe_current = probe_current
        elif spec.metadata.Acquisition_instrument.TEM.beam_current > 0:
            self.probe_current = spec.metadata\
                                     .Acquisition_instrument\
                                     .TEM.beam_current
        else:
            raise ValueError('Probe current is not defined')

        if beam_energy:
            self.beam_energy = beam_energy
        elif 'beam_energy' in spec.metadata\
                                  .Acquisition_instrument['TEM']\
                                  .keys():
            self.beam_energy = spec.metadata\
                                   .Acquisition_instrument['TEM']\
                                   .beam_energy
        else:
            raise ValueError('Beam energy is not defined')

        self.electron_dose = (self.probe_current * 1e-9 * self.live_time / 1.6e-19)
        if material == 'NiOx':
            self.elements = ['Ni', 'O']
            self.xray_lines = {'Ni_Ka': {'w': np.nan, 'sigma': np.nan},
                               'O_Ka': {'w': np.nan, 'sigma': np.nan}}
            self.get_xray_line_properties()
            self.density = 6.67
            self.density_sigma = np.nan
            if not thickness:
                self.thickness = 59
            if not thickness_sigma:
                self.thickness_sigma = 5
            self.composition_by_atom = {'Ni': {'atom_fraction': 0.5,
                                               'sigma': np.nan},
                                        'O': {'atom_fraction': 0.5,
                                              'sigma': np.nan}}
            self.molar_mass = self.get_molar_mass()
            self.composition_by_mass = self.at_to_wt()
            self.total_atoms_per_gram = self.get_atoms_per_gram()

        elif material == '2063a':
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

        elif material.split('_')[0] == 'Pure':
            element = material.split('_')[1]
            self.elements = [element, ]
            self.xray_lines = {element + '_Ka': {'w': np.nan, 'sigma': np.nan}}
            self.get_xray_line_properties()
            self.density = hs.material.elements[element]\
                                      .Physical_properties.density_gcm3
            self.density_sigma = np.nan
            if not thickness:
                self.thickness = None
            if not thickness_sigma:
                self.thickness_sigma = np.nan
            self.composition_by_atom = {element: {'atom_fraction': 1.0,
                                                  'sigma': np.nan}}
            self.composition_by_mass = {element: {'mass_fraction': 1.0,
                                                  'sigma': np.nan}}
            self.molar_mass = hs.material.elements[element]\
                                         .General_properties.atomic_weight
            self.total_atoms_per_gram = self.get_atoms_per_gram()

    def get_xray_line_properties(self):
        """
        Retrieve fundamental parameters for each line from database files.

        Includes ionization cross-sections and fluorescence yeild values for
        each line extracted from DTSA-2, available at:

        http://www.cstl.nist.gov/div837/837.02/epq/dtsa2/index.html

        Ionization cross-sections are from:

        D. Bote and F. Salvat, (2008) Calculations of inner-shell ionization
        by electron impact with the distorted-wave and planewave Born
        approximations. Phys Rev A 77, 042701.

        Fluorescence yield values are from ENDLIB97.  See

        D. E. Cullen, (1992) Program RELAX: A code designed to calculate X-ray
        and electron emission spectra as singly charged atoms relax back to
        neutrality. UCRL-ID-110438, Lawrence Livermore National Laboratory
        """
        if not self.xray_lines:
            raise ValueError('No X-ray lines defined!')
        fluor_yield = np.loadtxt(datapath + 'FluorescenceYield.txt')
        sigma = np.loadtxt(datapath + "AbsoluteIonizationCrossSection" + "BoteSalvat2008_KShell_%skeV.txt" %
                           str(self.beam_energy))
        for i in self.xray_lines:
            element = i.split('_')[0]
            atomic_number = hs.material.elements[element].General_properties.Z
            idx = np.where(fluor_yield[:, 0] == atomic_number)[0][0]
            self.xray_lines[i]['w'] = fluor_yield[idx, 1]
            self.xray_lines[i]['sigma'] = sigma[idx, 1]
        return

    def get_atoms_per_gram(self):
        """Calculate atomic number density per gram for the material."""
        total_atoms_per_gram = 0
        for i in self.composition_by_mass:
            total_atoms_per_gram +=\
                self.composition_by_mass[i]['mass_fraction']\
                / hs.material.elements[i].General_properties\
                                         .atomic_weight
        total_atoms_per_gram *= 6.02e23
        return total_atoms_per_gram

    def get_molar_mass(self):
        """Calculate the molar mass for the material."""
        molar_mass = 0
        for i in self.composition_by_atom:
            molar_mass +=\
                100 * self.composition_by_atom[i]['atom_fraction']\
                * hs.material.elements[i].General_properties\
                                         .atomic_weight
        return molar_mass

    def wt_to_at(self):
        """Convert composition by mass to composition by atom."""
        composition_by_atom = {}
        for i in self.composition_by_mass:
            atoms = (self.composition_by_mass[i]['mass_fraction'] / hs.material.elements[i].General_properties.atomic_weight * 6.02e23)
            atom_fraction = atoms / self.total_atoms_per_gram
            composition_by_atom[i] = {'atom_fraction': atom_fraction}
        return composition_by_atom

    def at_to_wt(self):
        """Convert composition by atom to composition by mass."""
        composition_by_mass = {}
        for i in self.composition_by_atom:
            mass = (100 * self.composition_by_atom[i]['atom_fraction'] * hs.material.elements[i].General_properties.atomic_weight)
            mass_fraction = mass / self.molar_mass
            composition_by_mass[i] = {'mass_fraction': mass_fraction,
                                      'sigma': np.nan}
        return composition_by_mass

    def get_atoms_per_volume(self, element):
        """Calculate the number of atoms per volume of an element in the material."""
        mass_frac = self.composition_by_mass[element]['mass_fraction']
        atomic_weight = hs.material.elements[element].General_properties.atomic_weight
        atoms_per_volume = (6.02e23 * mass_frac / atomic_weight * self.density)
        return atoms_per_volume

    def get_intensities(self, method='model', verbose=False,
                        plot_results=False):
        """Extract peak intensities."""
        spec = self.spec.deepcopy()
        spec.set_elements([])
        spec.set_lines([])

        if method == 'windows':
            if self.material == 'NiOx':
                spec.add_elements(['Fe', 'Ni', 'O', 'Mo', 'Si'])
                spec.add_lines(['Fe_Ka', 'Ni_Ka', 'Ni_Kb',
                                'Mo_Ka', 'O_Ka', 'Si_Ka',
                                'Mo_La', 'Ni_La'])
                lines_to_get = ['Fe_Ka', 'O_Ka', 'Ni_Ka',
                                'Ni_Kb', 'Mo_Ka', 'Mo_La']
                windows = np.array([[6.1, 6.2, 8.6, 8.7],
                                   [16.9, 17.1, 17.9, 18.0],
                                   [2.1, 2.2, 2.5, 2.6],
                                   [6.1, 6.2, 8.6, 8.7],
                                   [6.1, 6.2, 8.6, 8.7],
                                   [0.6, 0.7, 1.0, 1.1],
                                   [0.1, 0.2, 0.6, 0.7],
                                   [1.5, 1.6, 1.9, 2.0]])

            elif self.material == '2063a':
                spec.add_lines(['Mg_Ka', 'Si_Ka', 'Ca_Ka',
                                'Fe_Ka', 'O_Ka', 'Ar_Ka'])
                lines_to_get = ['Ar_Ka', 'C_Ka', 'Ca_Ka', 'Ca_Kb', 'Ca_La',
                                'Cu_Ka', 'Cu_Kb', 'Fe_Ka', 'Fe_Kb', 'Mg_Ka',
                                'O_Ka', 'Si_Ka']

                windows = np.array([[2.7, 2.8, 3.1, 3.2],
                                    [0.15, 0.19, 0.8, 0.9],
                                    [3.3, 3.45, 4.2, 4.4],
                                    [5.9, 6.1, 6.65, 6.75],
                                    [0.34, 0.41, 0.8, 0.9],
                                    [1.0, 1.1, 1.35, 1.42],
                                    [0.34, 0.41, 0.8, 0.9],
                                    [1.42, 1.55, 2.0, 2.12]])

            result = spec.\
                get_lines_intensity(background_windows=windows,
                                    plot_result=False)
            if verbose:
                print('Results for Window Method')
                print('Material: %s' % self.material)
                print('**********************')
                for i in result:
                    print('%s: %.2f counts' %
                          (i.metadata.Sample.xray_lines[0], i.data[0]))
                print('\n')

        elif method == 'model':
            if self.material == 'NiOx':
                spec = self.spec.isig[0.4:21.].deepcopy()
                spec.add_elements(['Fe', 'Ni', 'O', 'Mo', 'Si'])
                model = spec.create_model(auto_add_lines=False)
                model.add_family_lines()
                lines_to_get = ['Fe_Ka', 'O_Ka', 'Ni_Ka',
                                'Ni_Kb', 'Mo_Ka', 'Mo_La']

            elif self.material == '2063a':
                spec = self.spec.isig[:15.0].deepcopy()
                spec.add_elements(['C', 'Mg', 'Si', 'Ca',
                                   'Fe', 'O', 'Ar', 'Cu', ])
                spec.add_lines(['Mg_Ka', 'Si_Ka', 'Ca_Ka', 'Ca_Kb',
                                'Cu_Ka', 'Cu_Kb', 'Cu_La',
                                'Fe_Ka', 'Fe_Kb', 'Fe_La',
                                'O_Ka', 'Ar_Ka'])
                model = spec.create_model()
                model.free_xray_lines_width(['O_Ka', 'Fe_Ka'])
                model.free_xray_lines_energy(['O_Ka', 'Fe_Ka'])
                lines_to_get = ['Ar_Ka', 'Ca_Ka', 'Ca_Kb',
                                'Fe_Ka', 'Fe_Kb', 'Mg_Ka',
                                'O_Ka', 'Si_Ka']
                for i in model[1:]:
                    i.A.bmin = 0.0
            model.fit(bounded=True)
            model.fit_background()

            result = model.get_lines_intensity(plot_result=False,
                                               xray_lines=lines_to_get)
            if verbose:
                print('Results for Peak Fit')
                print('Material: %s' % self.material)
                print('**********************')
                for i in result:
                    print('%s: %.2f counts' %
                          (i.metadata.Sample.xray_lines[0], i.data))

                print('\nReduced Chi-Sq: %.2f\n' % model.red_chisq.data)
            if plot_results:
                model.plot(True)
                axis = plt.gca()
                labels = ['Data', 'Model', 'Background']
                axis.legend(labels)
                axis.set_ylim([-300, 1.1 * spec.data.max()])

        output = {}
        for i, _ in enumerate(result):
            line = result[i].metadata.Sample.xray_lines[0]
            if line in lines_to_get:
                output[line] = {'counts':
                                np.around(result[i].data[0], 2),
                                'uncertainty':
                                np.nan}
        self.intensities = output
        return

    def get_detector_characteristics(self, element=None, verbose=False):
        """
        Calculate detector characteristics from a spectrum of a known material.

        Args
        ----------
        spec : Hyperspy Signal1D, EDSSemSpectrum, or EDSTEMSpectrum
            XEDS spectrum collected from standard material.
        material : str
            Name of the standard material.  Must be either 'NiOx' or 2063a
        thickness : float or int
            Nominal thickness of standard material in nanometers.
            Default is 59 nm.
        live_time : float or int
            Spectrum acquisition time in seconds. Default is 200 seconds.
        tilt : float or int
            Specimen tilt in degrees.  Default is 0 degrees.
        thickness_error : float
            Error in nominal thickness measurement (+/- nanometers)
        probe_current : float
            Probe current in nanoamps. Default is 0.3 nA.

        Returns
        ----------
        results : Dict
            Dictionary containing all measured and calculated values.

        """
        if not self.thickness:
            raise ValueError('Specimen thickness not defined')

        if not self.probe_current:
            raise ValueError('Probe current is not defined')

        if not self.beam_energy:
            raise ValueError('Beam energy is not defined')

        if self.material == 'NiOx':
            element = 'Ni'
            xray_lines = ['Ni_Ka', 'Ni_Kb']
            lines = ['Ka', 'Kb']
            if self.intensities is None:
                self.get_intensities()
            counts = self.intensities['Ni_Ka']['counts']\
                + self.intensities['Ni_Kb']['counts']

        elif self.material == '2063a':
            if self.intensities is None:
                self.get_intensities()
            if not element:
                element = 'Fe'
            if element == 'Fe':
                xray_lines = ['Fe_Ka', 'Fe_Kb']
                lines = ['Ka', 'Kb']
                counts = self.intensities['Fe_Ka']['counts']\
                    + self.intensities['Fe_Kb']['counts']
            elif element == 'Si':
                xray_lines = ['Si_Ka', ]
                lines = ['Ka', ]
                counts = self.intensities['Si_Ka']['counts']
        else:
            raise ValueError('Unknown material')
        xray_energies = [None] * len(xray_lines)
        for i in range(0, len(xray_lines)):
            xray_energies[i] = hs.material.elements[element]\
                                 .Atomic_properties\
                                 .Xray_lines[lines[i]]\
                                 .energy_keV
        eff_thickness = self.thickness / np.cos(np.pi * self.specimen_tilt / 180)
        fluor_yield = self.xray_lines[xray_lines[0]]['w']
        sigma = self.xray_lines[xray_lines[0]]['sigma'] * 1e4
        n_atoms = self.get_atoms_per_volume(element) *\
            (eff_thickness * 1e-7)
        det_efficiency = counts /\
            (n_atoms * sigma * fluor_yield * self.electron_dose)
        omega = 4 * np.pi * det_efficiency
        omega = np.round(omega, 3)

        if verbose:
            print('Detector Solid-angle Calculation')
            print('*************************************')
            print('Beam energy (keV): %s' % str(self.beam_energy))
            print('Probe current (nA): %s' % str(self.probe_current))
            print('Live time (s): %s' % str(self.live_time))
            print('Electron dose: %.2e' % self.electron_dose)
            print('Nominal sample thickness (nm): %.1f' % self.thickness)
            print('Effective sample thickness (nm): %.1f\n' % eff_thickness)
            for i, _ in enumerate(xray_lines):
                print('X-ray line: %s @ %.2f keV' %
                      (xray_lines[i], xray_energies[0]))
            print('Counts detected: %s' % str(np.round(counts)))
            print('Ionization Cross-section (cm^2): %.2e' % sigma)
            print('Fluorescence Yield: %.3f' % fluor_yield)
            print('Atoms per Unit Area (cm^-2): %.2e\n' % n_atoms)
            print('Collection Efficiency: %.2f %%' % (100 * det_efficiency))
            print('Collection Solid-angle (srs): %.3f' % omega)

        return omega

    def calc_zeta_factor(self, plot_result=False, verbose=False):
        """
        Calculate Zeta factor from a spectrum collected from 2063a SRM.

        Args
        ------
        plot_result : bool
            If True, plot calculated Zeta factors as a function of
            X-ray energy.
        verbose : bool
            If True, print the results to the terminal
        """
        if self.intensities is None:
            self.get_intensities()

        rho = self.density * 1000
        rho_sigma = self.density_sigma * 1000
        eff_thickness = self.thickness\
            / np.cos(self.specimen_tilt * np.pi / 180)
        thickness_sigma = self.thickness_sigma
        dose = self.electron_dose

        if self.material == '2063a':
            elements = ['Mg', 'Si', 'Ca', 'Fe', 'O']

        elif self.material == 'NiOx':
            elements = ['Ni', 'O']

        lines = [i + '_Ka' for i in elements]

        zeta_factor_results = {}
        for i, _ in enumerate(elements):
            mass_fraction = (self.composition_by_mass[elements[i]]
                             ['mass_fraction'])
            uncertainty = self.composition_by_mass[elements[i]]['sigma']
            counts = self.intensities[lines[i]]['counts']
            zeta = rho * eff_thickness * 1e-9 * mass_fraction * dose / counts

            zeta_sigma = np.sqrt((uncertainty / mass_fraction)**2 + (2 * np.sqrt(counts) / counts)**2 + (thickness_sigma / eff_thickness)**2 + (rho_sigma / rho)**2) * zeta

            zeta_factor_results[lines[i]] = {'zeta_factor': zeta,
                                             'zeta_factor_sigma':
                                                 np.round(zeta_sigma, 2)}

        xray_energies = [hs.material.elements[i].Atomic_properties.
                         Xray_lines['Ka']['energy_keV'] for i in
                         elements]

        if verbose:
            print('Zeta Factor Analysis Results')
            print('Material: %s' % self.material)
            print('*************************************')
            print('Beam energy (keV): %s' % str(self.beam_energy))
            print('Probe current (nA): %s' % str(self.probe_current))
            print('Live time (s): %s' % str(self.live_time))
            print('Electron dose: %.2e' % self.electron_dose)
            print('Nominal sample thickness (nm): %.1f' % self.thickness)
            print('Effective sample thickness (nm): %.1f\n' % eff_thickness)
            count = 0
            for i in zeta_factor_results:
                print('%s (%.2f keV): %.2f +/- %.2f' %
                      (i, xray_energies[count],
                       zeta_factor_results[i]['zeta_factor'],
                       zeta_factor_results[i]['zeta_factor_sigma']))
                count += 1

        if plot_result:
            plt.figure()
            plt.scatter(xray_energies,
                        [zeta_factor_results[i]['zeta_factor']
                         for i in zeta_factor_results])
            plt.xlabel('X-ray Energy (keV)')
            plt.ylabel(r'$\zeta$ factor (kg-electron/(m$^{2}$-photon))')

        self.zeta_factors = zeta_factor_results
        return
