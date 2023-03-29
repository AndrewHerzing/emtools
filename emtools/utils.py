# -*- coding: utf-8 -*-
#
# This file is part of EMTools.

"""
Utilities module for EMTools package.

@author: Andrew Herzing
"""

import numpy as np
import hyperspy.api as hs
from scipy.signal import convolve2d

e = 1.602e-19       # Charge of electron (Coulombs)
m0 = 9.109e-31      # Rest mass of electron (kg)
m0c2 = 511          # Rest energy of electron (keV)
h = 6.626e-34       # Planck's constant
c = 2.998e8         # Speed of light in vacuum (m/s)
Na = 6.0221409e23   # Avogadro's number


def mrads_to_hkl(angle, voltage):
    """
    Convert from an diffraction angle (mrads) to lattice spacing (nm).

    Args
    ----------
    mrads : float
        Scattering angle in mrads
    voltage : float or int
        Electron beam voltage (kV)

    Returns
    ----------
    d : float
        Lattice spacing in nanometers
    """
    wavelength = voltage_to_wavelength(300, True)
    d = wavelength / (2 * np.sin(angle / 1000))
    return d


def mrads_to_k(angle, voltage):
    """
    Convert from an angular measurement (mrads) to reciprocal space (nm^-1).

    Args
    ----------
    mrads : float
        Scattering angle in mrads
    voltage : float or int
        Electron beam voltage (kV)

    Returns
    ----------
    k : float
        Reciprocal lattice spacing in either inverse nanometers
    """
    wavelength = voltage_to_wavelength(voltage, True)
    k = (2 * np.sin(angle / 1000)) / wavelength
    return k


def k_to_mrads(k, voltage):
    """
    Convert from a reciprocal space (nm^-1) value an angular value (mrads).

    Args
    ----------
    k : float
        Reciprocal lattice spacing in either inverse nanometers
    voltage : float or int
        Electron beam voltage (kV)

    Returns
    ----------
    angle : float
        Scattering angle in mrads
    """
    wavelength = voltage_to_wavelength(voltage, True)
    angle = 1000 * np.arcsin(k * wavelength / 2)
    return angle


def voltage_to_wavelength(voltage, relativistic=False):
    """
    Calculate electron wavelength given voltage.

    Args
    ----------
    voltage : float
        Accelerating voltage (in kV)

    relativistic : bool
        If True, calculate the relatistically corrected wavelength

    Returns
    ----------
    wavelength : float
        Calculated wavelength (in nanometers)
    """
    if relativistic:
        correction = (1 + ((e * voltage * 1000) / (2 * m0 * c**2)))
        wavelength = h / np.sqrt(2 * m0 * e * voltage * 1000 * correction)
    else:
        wavelength = h / np.sqrt(2 * m0 * e * voltage * 1000)
    return wavelength * 1e9


def get_relativistic_mass(voltage):
    """
    Calculate relativistic mass given voltage.

    Args
    ----------
    voltage : float
        Accelerating voltage (in kV)

    Returns
    ----------
    rel_mass : float
        Calculated relativistic mass (in kg)
    """
    rel_mass = (1 + voltage / m0c2) * m0
    return rel_mass


def get_atom_mass(element):
    """
    Calculate the mass of an atom of a given element.

    Args
    ----------
    element : string
        Elemental symbol

    Returns
    ----------
    atom_mass : float
        Mass of an atom of element (in kg)
    """
    atomic_weight =\
        hs.material.elements[element].General_properties.atomic_weight
    atom_mass = atomic_weight / Na / 1000
    return atom_mass


def calc_rutherford_energy_loss(energy, element, theta, relativistic=False):
    """
    Calculate the energy loss due to Rutherford scattering.

    Uses the impulse approximation as calculated in:

    Lovejoy et al., M&M, 20(S3) (2014) 558.
    DOI: 10.1017/S1431927614004516
    Args
    ----------
    energy : float
        Incident electron energy in keV
    element : string
        Elemental symbol of scattering nucleus
    theta : float or array
        Scattering angle in degrees
    relativistic : bool
        If True, apply relativistic correction to electron mass

    Returns
    ----------
    energy_loss : float or array
        Energy lost due to scattering to angle theta (in eV)
    """
    theta = 1000 * theta * np.pi / 180
    atom_mass = get_atom_mass(element)
    if relativistic:
        me = get_relativistic_mass(energy)
        mass_ratio = me / atom_mass
    else:
        mass_ratio = m0 / atom_mass

    energy_loss = 4 * np.sin(theta / 2000)**2 * mass_ratio * energy * 1000
    return energy_loss


def calc_ERBS_energy_loss(energy, element, theta, relativistic=False):
    """
    Calculate the energy loss due to Rutherford backscattering.

    Uses the equation in:

    Vos, Ultramicroscopy, 92 (2002) 143.

    Args
    ----------
    energy : float
        Incident electron energy in keV
    element : string
        Elemental symbol of scattering nucleus
    theta : float or array
        Scattering angle in degrees
    relativistic : bool
        If True, apply relativistic correction to electron mass

    Returns
    ----------
    energy_loss : float or array
        Energy lost due to scattering to angle theta (in eV)
    """
    theta = 1000 * theta * np.pi / 180
    atom_mass = get_atom_mass(element)
    if relativistic:
        me = get_relativistic_mass(energy)
        k = np.sqrt(energy * 1000 * 2 * me)
    else:
        k = np.sqrt(energy * 1000 * 2 * m0)

    energy_loss = (2 / atom_mass) * (k * np.sin(theta / 2000))**2
    return energy_loss


def calc_probe_size(ht, alpha, relativistic=True):
    """
    Calculate probe size given voltage and convergence angle.

    Args
    ----------
    ht : float
        Accelerating voltage (in kV)

    alpha : float
        Convergence semi-angle (in mrads)

    relativistic : bool
        If True, use the relativistically corrected wavelength for the
        calculation

    Returns
    ----------
    probe_size : float
        Calculated probe diameter (in nanometers)
    """
    wavelength = voltage_to_wavelength(ht, relativistic)
    probe_size = 0.61 * wavelength / alpha / 1000
    return probe_size


def change_units(im, new_units='nm'):
    """
    Change the spatial calibration units of an image.

    Args
    ----------
    im : Hyperspy Signal2D
        Image to change units
    new_units : string
        New units. Must be 'nm', 'um', or 'A'.

    Returns
    ----------
    im_changed : Hyperspy Signal2D
        Copy of input image with units changed.
    """
    if im.axes_manager[0].units == new_units:
        return im
    elif new_units == 'A':
        if im.axes_manager[0].units in ['um', 'µm', '\u03BCm', '\xb5m']:
            im.axes_manager[0].units = '\u00c5'
            im.axes_manager[0].scale = 1e4 * im.axes_manager[0].scale
            im.axes_manager[1].units = '\u00c5'
            im.axes_manager[1].scale = 1e4 * im.axes_manager[1].scale
        elif im.axes_manager[0].units == 'nm':
            im.axes_manager[0].units = '\u00c5'
            im.axes_manager[0].scale = 10 * im.axes_manager[0].scale
            im.axes_manager[1].units = '\u00c5'
            im.axes_manager[1].scale = 10 * im.axes_manager[1].scale
    elif new_units == 'nm':
        if im.axes_manager[0].units in ['um', 'µm', '\u03BCm', '\xb5m']:
            im.axes_manager[0].units = 'nm'
            im.axes_manager[0].scale = 1e3 * im.axes_manager[0].scale
            im.axes_manager[1].units = 'nm'
            im.axes_manager[1].scale = 1e3 * im.axes_manager[1].scale
        elif im.axes_manager[0].units in ['A', '\u00c5']:
            im.axes_manager[0].units = 'nm'
            im.axes_manager[0].scale = im.axes_manager[0].scale / 10
            im.axes_manager[1].units = 'nm'
            im.axes_manager[1].scale = im.axes_manager[1].scale / 10
    return im


def calc_mean_z(composition, elements, comp_type='at'):
    """
    Calculate the mean atomic number given composition and elements.

    Args
    ----------
    composition : list or NumPy array
        Composition of material in percent
    elements : list
        List of strings giving the elements in same order as composition
    comp_type : str
        Either 'at' or 'wt' to indicate type of composition provided.  If 'wt',
        it will be converted to 'at' for the calculation

    Returns
    ----------
    Zmean : float
        Mean atomic number

    """
    Zs = np.array([hs.material.elements[i].General_properties.Z for i in elements])
    if comp_type == 'at':
        atom_fracs = composition
        Zmean = np.sum(Zs * atom_fracs / 100)
    elif comp_type == 'wt':
        atom_fracs = hs.material.weight_to_atomic(composition, elements)
        Zmean = np.sum(Zs * atom_fracs / 100)
    return Zmean


def estimate_image_noise(image):
    """
    Estimates variance of Gaussian noise in an image.

    Uses the method reported in:

    J. Immerkaer, Computer Vision and Image Understanding, 64 (1996) 300.

    """
    H, W = image.shape

    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(image, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
    return sigma


def calc_depth_of_focus(ht, alpha, relativistic=True):
    """
    Estimate the depth of focus for given accelerating voltage and convergence semi-angle.

    Args
    ----------
    ht : float
        Microscope accelerating voltage in kV.
    alpha : float
        Convergence semi-angle in mrads.
    relativistic : bool
        If True, use the relativistic wavelength.  Default is True

    Returns
    ----------
    dof : float
        Estimated depth of focus in nanometers.
    """
    wavelength = voltage_to_wavelength(ht, relativistic)
    dof = 0.61 * wavelength / alpha
    return dof


def calc_dose_pixel(probe_current, pixel_size, dwell_time):
    """
    Calculate electron dose given probe current, dwell time, and pixel size.

    Args
    ---------
    probe_current : float
        Probe current in amps

    dwell_time : float
        Dwell time per pixel in seconds

    pixel_size : float
        Pixel size in nanometers

    Returns
    ----------
    dose : float
        Calculated electron dose in electrons per square nanometer
    """
    dose = 6.242e18 * probe_current * dwell_time / pixel_size**2
    return dose


def calc_dose_probe(probe_current, probe_fwhm, dwell_time):
    """
    Calulate electron based on the measured probe size.

    Args
    ---------
    probe_current : float
        Measured probe current in amps
    probe_fwhm : float
        Measured probe FWHM in nanometers
    dwell_time : float
        Per pixel dwell time in seconds

    Returns
    ---------
    dose : float
        Calculated electron dose in electrons per square nanometer

    """
    n_electrons = 6.242e18 * probe_current * dwell_time
    dose = n_electrons / (np.pi * (probe_fwhm / 2)**2)
    return dose
