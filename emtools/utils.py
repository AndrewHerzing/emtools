import numpy as np
import hyperspy.api as hs

e = 1.602e-19       # Charge of electron (Coulombs)
m0 = 9.109e-31      # Rest mass of electron (kg)
m0c2 = 511          # Rest energy of electron (keV)
h = 6.626e-34       # Planck's constant
c = 2.998e8         # Speed of light in vacuum (m/s)
Na = 6.0221409e23   # Avogadro's number


def voltage_to_wavelength(voltage, relativistic=False):
    """
    Calculates electron wavelength given voltage

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
    Calculates relativistic mass given voltage

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
    Calculate the mass of an atom of a given element

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
    Calculate the energy loss due to Rutherford scattering

    Uses the impulse approximation as calculated in:

    Lovejoy et al., M&M, 20(S3) (2014) 558.
    DOI: 10.1017/S1431927614004516
    Args
    ----------
    energy : float
        Incident electron energy (in keV)
    element : string
        Elemental symbol of scattering nucleus
    theta : float or array
        Scattering angle (in mrads)
    relativistic : bool
        If True, apply relativistic correction to electron mass

    Returns
    ----------
    energy_loss : float or array
        Energy lost due to scattering to angle theta (in eV)
    """

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
    Calculate the energy loss due to Rutherford backscattering

    Uses the equation in:

    Vos, Ultramicroscopy, 92 (2002) 143.

    Args
    ----------
    energy : float
        Incident electron energy (in keV)
    element : string
        Elemental symbol of scattering nucleus
    theta : float or array
        Scattering angle (in mrads)
    relativistic : bool
        If True, apply relativistic correction to electron mass

    Returns
    ----------
    energy_loss : float or array
        Energy lost due to scattering to angle theta (in eV)
    """

    atom_mass = get_atom_mass(element)
    if relativistic:
        me = get_relativistic_mass(energy)
        k = np.sqrt(energy * 1000 * 2 * me)
    else:
        k = np.sqrt(energy * 1000 * 2 * m0)

    energy_loss = (2 / atom_mass) * (k * np.sin(theta / 2000))**2
    return energy_loss
