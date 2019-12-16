# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
EDS module for EMTools package

@author: Andrew Herzing
"""

from matplotlib import pylab as plt
import hyperspy.api as hs
import numpy as np
import pprint as pp
import imp

datapath = imp.find_module("emtools")[1] + "/data/"


def get_test_spectrum(material='2063a'):
    allowed_mats = ['2063a', 'niox']

    if material.lower() not in allowed_mats:
        raise ValueError('Unknown material. Must be one of %s.' %
                         ', '.join(allowed_mats))

    s = hs.load(datapath + 'XEDS_' + material + '.hdf5')
    return s


def get_detector_efficiency(detector_name):
    """
    Reads detecor efficiency from database

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

    detector_efficiency = np.loadtxt(datapath +
                                     detector_name +
                                     '_DetectorEfficiencyCurve.txt')
    detector_efficiency = hs.signals.Signal1D(detector_efficiency)
    detector_efficiency.axes_manager[0].name = 'Energy'
    detector_efficiency.axes_manager[0].units = 'keV'
    detector_efficiency.axes_manager[0].scale = 0.01
    return detector_efficiency


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


class QuantSpec:
    """
    Class to create materials with provided composition

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

    """
    def __init__(self, spec, material, beam_energy=None, thickness=None,
                 thickness_sigma=None, live_time=None, probe_current=None,
                 specimen_tilt=None):
        """
        Constructor for Material class.

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

        self.electron_dose = (self.probe_current * 1e-9
                              * self.live_time / 1.6e-19)
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

    def get_xray_line_properties(self):
        """
        Retrieves fundamental parameters for each line from database files.

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
        """
        Calculates atomic number density per gram for the material.

        """
        total_atoms_per_gram = 0
        for i in self.composition_by_mass:
            total_atoms_per_gram +=\
                self.composition_by_mass[i]['mass_fraction']\
                / hs.material.elements[i].General_properties\
                                         .atomic_weight
        total_atoms_per_gram *= 6.02e23
        return total_atoms_per_gram

    def get_molar_mass(self):
        """
        Calculates the molar mass for the material.

        """
        molar_mass = 0
        for i in self.composition_by_atom:
            molar_mass +=\
                100 * self.composition_by_atom[i]['atom_fraction']\
                * hs.material.elements[i].General_properties\
                                         .atomic_weight
        return molar_mass

    def wt_to_at(self):
        """
        Converts composition by mass to composition by atom.

        """
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
        """
        Converts composition by atom to composition by mass.

        """
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
        """
        Calculates the number of atoms per volume of an element in the material

        """
        mass_frac = self.composition_by_mass[element]['mass_fraction']
        atomic_weight = hs.material.elements[element]\
            .General_properties\
            .atomic_weight
        atoms_per_volume = (6.02e23
                            * mass_frac / atomic_weight
                            * self.density)
        return atoms_per_volume

    def get_intensities(self, method='model', verbose=False,
                        plot_results=False):
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
                bw = np.array([[6.1, 6.2, 8.6, 8.7],
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

                bw = np.array([[2.7, 2.8, 3.1, 3.2],
                               [0.15, 0.19, 0.8, 0.9],
                               [3.3, 3.45, 4.2, 4.4],
                               [5.9, 6.1, 6.65, 6.75],
                               [0.34, 0.41, 0.8, 0.9],
                               [1.0, 1.1, 1.35, 1.42],
                               [0.34, 0.41, 0.8, 0.9],
                               [1.42, 1.55, 2.0, 2.12]])

            result = spec.\
                get_lines_intensity(background_windows=bw,
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
                m = spec.create_model(auto_add_lines=False)
                m.add_family_lines()
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
                m = spec.create_model()
                m.free_xray_lines_width(['O_Ka', 'Fe_Ka'])
                m.free_xray_lines_energy(['O_Ka', 'Fe_Ka'])
                lines_to_get = ['Ar_Ka', 'Ca_Ka', 'Ca_Kb',
                                'Fe_Ka', 'Fe_Kb', 'Mg_Ka',
                                'O_Ka', 'Si_Ka']
                for i in m[1:]:
                    i.A.bmin = 0.0
            m.fit(bounded=True)
            m.fit_background()

            result = m.get_lines_intensity(plot_result=False,
                                           xray_lines=lines_to_get)
            if verbose:
                print('Results for Peak Fit')
                print('Material: %s' % self.material)
                print('**********************')
                for i in result:
                    print('%s: %.2f counts' %
                          (i.metadata.Sample.xray_lines[0], i.data))

                print('\nReduced Chi-Sq: %.2f\n' % m.red_chisq.data)
            if plot_results:
                m.plot(True)
                ax = plt.gca()
                labels = ['Data', 'Model', 'Background']
                ax.legend(labels)
                ax.set_ylim([-300, 1.1 * spec.data.max()])

        output = {}
        for i in range(0, len(result)):
            line = result[i].metadata.Sample.xray_lines[0]
            if line in lines_to_get:
                output[line] = {'counts':
                                np.around(result[i].data[0], 2),
                                'uncertainty':
                                np.nan}
        self.intensities = output
        return

    def get_detector_characteristics(self, element=None, display=True,
                                     verbose=False):
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
        display : bool
            If True, print the results to the terminal.

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
            if self.intensites is None:
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
        eff_thickness = self.thickness / np.cos(np.pi
                                                * self.specimen_tilt
                                                / 180)
        w = self.xray_lines[xray_lines[0]]['w']
        sigma = self.xray_lines[xray_lines[0]]['sigma'] * 1e4
        N_atoms = self.get_atoms_per_volume(element) *\
            (eff_thickness * 1e-7)
        nu = counts / (N_atoms * sigma * w * self.electron_dose)
        omega = 4 * np.pi * nu
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
            for i in range(0, len(xray_lines)):
                print('X-ray line: %s @ %.2f keV' %
                      (xray_lines[i], xray_energies[0]))
            print('Counts detected: %s' % str(np.round(counts)))
            print('Ionization Cross-section (cm^2): %.2e' % sigma)
            print('Fluorescence Yield: %.3f' % w)
            print('Atoms per Unit Area (cm^-2): %.2e\n' % N_atoms)
            print('Collection Efficiency: %.2f %%' % (100 * nu))
            print('Collection Solid-angle (srs): %.3f' % omega)

        return omega

    def calc_zeta_factor(self, plot_result=False, verbose=False):
        """
        Calculate Zeta factor from a spectrum collected from 2063a SRM

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

        rho = self.density
        rho_sigma = self.density_sigma
        thickness = self.thickness / np.cos(self.specimen_tilt * np.pi / 180)
        thickness_sigma = self.thickness_sigma
        dose = self.electron_dose

        mass_fraction_mg = self.composition_by_mass['Mg']['mass_fraction']
        uncertainty_mg = self.composition_by_mass['Mg']['sigma']
        zeta_mg = rho * thickness * mass_fraction_mg * \
            dose / self.intensities['Mg_Ka']['counts']
        zeta_mg_sigma = np.sqrt((uncertainty_mg /
                                mass_fraction_mg)**2 +
                                (2 *
                                np.sqrt(self.intensities['Mg_Ka']['counts']) /
                                self.intensities
                                ['Mg_Ka']['counts'])**2 +
                                (thickness_sigma / thickness)**2 +
                                (rho_sigma / rho)**2) * zeta_mg

        mass_fraction_si = self.composition_by_mass['Si']['mass_fraction']
        uncertainty_si = self.composition_by_mass['Si']['sigma']
        zeta_si = rho * thickness * mass_fraction_si * \
            dose / self.intensities['Si_Ka']['counts']
        zeta_si_sigma = np.sqrt((uncertainty_si /
                                mass_fraction_si)**2 +
                                (2 *
                                np.sqrt(self.intensities['Si_Ka']['counts']) /
                                self.intensities['Si_Ka']['counts'])**2 +
                                (thickness_sigma / thickness)**2 +
                                (rho_sigma / rho)**2) * zeta_si

        mass_fraction_ca = self.composition_by_mass['Ca']['mass_fraction']
        uncertainty_ca = self.composition_by_mass['Ca']['sigma']
        zeta_ca = rho * thickness * mass_fraction_ca * \
            dose / self.intensities['Ca_Ka']['counts']
        zeta_ca_sigma = np.sqrt((uncertainty_ca /
                                mass_fraction_ca)**2 +
                                (2 *
                                np.sqrt(self.intensities['Ca_Ka']['counts']) /
                                self.intensities['Ca_Ka']['counts'])**2 +
                                (thickness_sigma / thickness)**2 +
                                (rho_sigma / rho)**2) * zeta_ca

        mass_fraction_fe = self.composition_by_mass['Fe']['mass_fraction']
        uncertainty_fe = self.composition_by_mass['Fe']['sigma']
        zeta_fe = rho * thickness * mass_fraction_fe * \
            dose / self.intensities['Fe_Ka']['counts']
        zeta_fe_sigma = np.sqrt((uncertainty_fe /
                                mass_fraction_fe)**2 +
                                (2 *
                                np.sqrt(self.intensities['Fe_Ka']['counts']) /
                                self.intensities['Fe_Ka']['counts'])**2 +
                                (thickness_sigma / thickness)**2 +
                                (rho_sigma / rho)**2) * zeta_fe

        mass_fraction_o = self.composition_by_mass['O']['mass_fraction']
        uncertainty_o = self.composition_by_mass['O']['sigma']
        zeta_o = rho * thickness * mass_fraction_o * \
            dose / self.intensities['O_Ka']['counts']
        zeta_o_sigma = np.sqrt((uncertainty_o /
                                mass_fraction_o)**2 +
                               (2 *
                               np.sqrt(self.intensities['O_Ka']['counts']) /
                                self.intensities['O_Ka']['counts'])**2 +
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
                                  'zeta_factor_sigma':
                                  np.round(zeta_mg_sigma, 2)},
                        'Si_Ka': {'zeta_factor': np.round(zeta_si, 2),
                                  'zeta_factor_sigma':
                                  np.round(zeta_si_sigma, 2)},
                        'Ca_Ka': {'zeta_factor': np.round(zeta_ca, 2),
                                  'zeta_factor_sigma':
                                  np.round(zeta_ca_sigma, 2)},
                        'Fe_Ka': {'zeta_factor':
                                  np.round(zeta_fe, 2),
                                  'zeta_factor_sigma':
                                  np.round(zeta_fe_sigma, 2)},
                        'O_Ka': {'zeta_factor': np.round(zeta_o, 2),
                                 'zeta_factor_sigma':
                                 np.round(zeta_o_sigma, 2)}
                        }
        if verbose:
            pp.pprint(zeta_factors)
        return zeta_factors
