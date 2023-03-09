# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
EELS module for EMTools package

@author: Andrew Herzing
"""

import imp
import numpy as np
from scipy.integrate import quad
import scipy
import matplotlib.pylab as plt
import hyperspy.api as hs
from hyperspy.misc.eels.electron_inelastic_mean_free_path import iMFP_Iakoubovskii, iMFP_angular_correction

def fitsigmatotal(energy, sigma, line=None, plot_results=False):
    """
    Determine total cross-section via asymptotic fitting

    Args
    ----------
    energy : NumPy array
        Energy values to fit
    sigma : NumPy array
        Values of cross-section to fit
    line : str
        Name of Shell to use for Label
    plot_results : bool
        If True, plot the results of the fit

    Returns
    ----------
    fit_data[2] : Y-component of the fitting result

    """
    def asymptotic(x, a, b, c):
        return a / (x - b) + c

    fit_data, _ = scipy.optimize.curve_fit(asymptotic,
                                           energy,
                                           sigma,
                                           (10., 1., 1.))
    if plot_results:
        _, ax = plt.subplots(1, figsize=(10, 6))
        ax.plot(energy, sigma, '-o', color='red')
        ax.axhline(fit_data[2], linestyle='--', color='black')
        _ = ax.set_ylim(0, 1.1 * fit_data[2])
        _ = ax.set_xlim(0, 1.1 * energy[-1])
        _ = ax.set_title('%s Total Cross Section' % line)
        _ = ax.text(0.6 * energy[-1], 0.1 * fit_data[2],
                    ('Fitted Total Sigma: %.2f' % fit_data[2]), fontsize=12)
    return fit_data


def sigmak(z=None, ek=None, delta=None, e0=None, beta=None, verbose=True):
    """
    Calculate the K-shell ionization cross-section as a function of energy

    Python implementation of Matlab code from R. F. Egerton at:
    http://www.tem-eels.ca/egerton-laser/programs/SIGMAKL-instructions.htm

    Uses relativistic kinematics and a hydrogenic model with inner-shell
    screening constant of 0.5. Details in R.F.Egerton: EELS in the Electron
    Microscope, 3rd edition, Springer 2011.

    Given the atomic number

    Args
    ----------
    z : int
        Atomic number of the element of interest
    ek : float
        K-shell ionization energy, in eV
    delta1 : float
        Energy increment of output data, in eV
    e0 : float
        Incident electron energy (keV)
    beta : float
        Maximum scattering angle contributing to the cross-section (mrads)
    verbose : bool
        If True, output results to the terminal

    Returns
    ----------
    eout : NumPy array
        Energy values for calculated cross-section (eV)
    sigmaout : Numpy array
        Calculated cross-section values (cm^2 per atom)

    """

    if verbose:
        print('---------------Sigma-K----------------\n')
        print('Atomic number Z : %s' % z)
        print('K-edge threshold energy, ek (eV) : %s' % ek)
        print('Integration window, delta (eV) : %s' % delta)
        print('Incident-electron energy E0(keV) : %s' % e0)
        print('Collection semi-angle beta(mrad) : %s' % beta)

    einc = delta / 10

    r = 13.606
    e = ek
    b = beta / 1000
    t = 511060 * (1 - 1 / (1 + e0 / (511.06)) ** 2) / 2
    gg = 1 + e0 / 511.06
    p02 = t / r / (1 - 2 * t / 511060)
    f = 0
    s = 0
    sigma = 0
    #     integrate over scattering angle FOR EACH ENERGY LOSS:
    dsbdep = 0
    dfprev = 0
    eout = []
    sigmaout = []
    if verbose:
        print('\nE(eV)    ds/dE(barn/eV)  Delta(eV)   Sigma(barn)     f(0)')
    for j in range(0, 50):
        qa021 = e ** 2 / (4 * r * t) + e ** 3 / (8 * r * t ** 2 * gg ** 3)
        pp2 = p02 - e / r * (gg - e / 1022120)
        qa02m = qa021 + 4 * np.sqrt(p02 * pp2) * (np.sin(b / 2)) ** 2

        #   dsbyde IS THE ENERGY-DIFFERENTIAL X-SECN (barn/eV/atom)
        dsbyde = 3.5166e8 * (r / t) * (r / e) * \
            quad(lambda x: gos_k(e, np.exp(x), z),
                 np.log(qa021), np.log(qa02m))[0]
        dfdipl = gos_k(e, qa021, z)  # dipole value
        delta_current = e - ek

        if j != 0:
            s = np.log(dsbdep / dsbyde) / np.log(e / (e - einc))
            sginc = (e * dsbyde - (e - einc) * dsbdep) / (1 - s)
            sigma = sigma + sginc        # barn/atom
            f = f + (dfdipl + dfprev) / 2 * einc

        if verbose:
            print('%4g %17.6f %10d %13.2f %8.4f' %
                  (e, dsbyde, delta_current, sigma, f))
        eout.append(e)
        sigmaout.append(sigma)
        if einc == 0:
            if verbose:
                print('\nEnergy increment fell to zero')
            break

        if delta_current >= delta:
            if sginc < 0.0001 * sigma:
                if verbose:
                    print('\nChange in sigma less than 0.0001')
                break

            einc = einc * 2

        e = e + einc
        if e > t:
            if verbose:
                print('\nEnergy threshold exceeded')
            break

        dfprev = dfdipl
        dsbdep = dsbyde
    if verbose:
        print('%s iterations completed' % str(j))
    eout = np.array(eout)
    sigmaout = np.array(sigmaout)
    return eout, sigmaout


def gos_k(E, qa02, z):
    """
    Calculate DF/DE (per eV and per atom) for K-shell

    Note: quad function only works with qa02 due to IF statements in function

    Args
    ----------
    E : int
        Energy of incident electron (keV)
    qa02 : float

    z : int
        Atomic number of scattering atom

    """
    # global r
    if (not np.isscalar(E) or not np.isscalar(z)):
        print('gosfunc: E and z input parameters must be scalar')

    r = 13.606
    zs = 1.0
    rnk = 1
    if z != 1:
        zs = z - 0.50
        rnk = 2

    q = qa02 / zs ** 2
    kh2 = E / r / zs ** 2 - 1
    akh = np.sqrt(np.abs(kh2))
    if akh <= 0.1:
        akh = 0.1

    if kh2 >= 0.0:
        d = 1 - np.exp(-2 * np.pi / akh)
        bp = np.arctan(2 * akh / (q - kh2 + 1))
        if bp < 0:
            bp = bp + np.pi

        c = np.exp((-2 / akh) * bp)
    else:
        # SUM OVER EQUIVALENT BOUND STATES:
        d = 1
        y = (-1 / akh * np.log((q + 1 - kh2 + 2 * akh) /
                               (q + 1 - kh2 - 2 * akh)))
        c = np.exp(y)

    a = ((q - kh2 + 1) ** 2 + 4. * kh2) ** 3
    out = 128 * rnk * E / r / zs ** 4. * c / d * (q + kh2 / 3 + 1 / 3) / a / r

    return out


def sigmal(z=None, delta=None, e0=None, beta=None, verbose=True):
    """
    Calculate the L-shell ionization cross-scetion as function of energy

    Python implementation of Matlab code from R. F. Egerton at:
    http://www.tem-eels.ca/

    Uses relativistic kinematics and a modified hydrogenic model with
    inner-shell screening constant of 0.5. See Egerton, Proc. EMSA (1981)
    p.198 & 'EELS in the TEM'. The GOS is reduced by a screening factor RF,
    based on data from several sources.
    See Ultramicroscopy 50 (1993) p. 22.

    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition,
    Springer 2011

    Args
    ----------
    z : int
        Atomic number of the element of interest
    delta : float
        Energy increment of output data, in eV
    e0 : float
        Incident electron energy (keV)
    beta : float
        Maximum scattering angle contributing tot the cross-section (mrads)

    Returns
    ----------
    eout : NumPy array
        Energy values for calculated cross-section (eV)
    sigmaout : Numpy array
        Calculated cross-section values (cm^2 per atom)

    """

    IE3 = [73, 99, 135, 164, 200, 245, 294, 347, 402, 455, 513, 575,
           641, 710, 779, 855, 931, 1021, 1115, 1217, 1323, 1436, 1550, 1675]

    if verbose:
        print('\n----------------Sigma-L---------------\n')
        print('Atomic number Z : %s' % z)
        print('Integration window Delta (eV) : %s' % delta)
        print('Incident-electron energy E0(keV) : %s' % e0)
        print('Collection semi-angle Beta(mrad) : %s\n' % beta)
        print('E(eV)    ds/dE(barn/eV)  Delta(eV)   Sigma(barn)     f(0)')

    einc = delta / 10
    r = 13.606
    iz = np.int(np.fix(z) - 13)
    el3 = (IE3[iz])

    e = el3
    b = beta / 1000
    t = 511060 * (1 - 1 / (1 + e0 / (511.06))**2) / 2
    gg = 1 + e0 / 511.06
    p02 = t / r / (1 - 2 * t / 511060)
    f = 0
    s = 0
    sigma = 0
    dsbdep = 0
    dfprev = 0

    eout = []
    sigmaout = []
    #     CALCULATE cross sections FOR EACH ENERGY LOSS:
    for j in range(0, 40):
        qa021 = e**2 / (4 * t * r) + e**3 / (8 * r * t**2 * gg**3)
        pp2 = p02 - e / r * (gg - e / 1022120)
        qa02m = qa021 + 4 * np.sqrt(p02 * pp2) * (np.sin(b / 2))**2

        dsbyde = 3.5166e8 * (r / t) * (r / e) *\
            quad(lambda x: gos_l(e, np.exp(x), z),
                 np.log(qa021), np.log(qa02m))[0]

        dfdipl = gos_l(e, qa021, z)  # dipole value

        delta_current = e - el3
        if j != 0:
            s = np.log(dsbdep / dsbyde) / np.log(e / (e - einc))
            sginc = (e * dsbyde - (e - einc) * dsbdep) / (1 - s)
            # sigma is the EELS cross section cm**2 per atom
            sigma = sigma + sginc
            f = f + (dfdipl + dfprev) * einc / 2
            if delta_current >= 10:
                if verbose:
                    print('%4g %17.6f %10d %13.2f %8.4f' %
                          (e, dsbyde, delta_current, sigma, f))
                eout.append(e)
                sigmaout.append(sigma)

        if delta_current >= delta:
            if sginc < 0.001 * sigma:
                break
            einc = einc * 2

        e = e + einc
        if e > t:
            e = e - einc
            break

        dfprev = dfdipl
        dsbdep = dsbyde

    eout = np.array(eout)
    sigmaout = np.array(sigmaout)
    return eout, sigmaout


def gos_l(E, qa02, z):
    """
    Calculate DF/DE (per eV and per atom) for L-shell

    Note: quad function only works with qa02 due to IF statements in function

    Args
    ----------
    E : int
        Energy of incident electron (keV)
    qa02 : float

    z : int
        Atomic number of scattering atom

    """
    IE3 = [73, 99, 135, 164, 200, 245, 294, 347, 402, 455, 513, 575,
           641, 710, 779, 855, 931, 1021, 1115, 1217, 1323, 1436, 1550, 1675]
    XU = [.52, .42, .30, .29, .22, .30, .22, .16, .12, .13, .13, .14,
          .16, .18, .19, .22, .14, .11, .12, .12, .12, .10, .10, .10]
    IE1 = [118, 149, 189, 229, 270, 320, 377, 438, 500, 564, 628, 695,
           769, 846, 926, 1008, 1096, 1194, 1142, 1248, 1359, 1476, 1596, 1727]

    if(not np.isscalar(E) or not np.isscalar(z)):
        raise ValueError('gosfunc: E and z input parameters must be scalar')

    r = 13.606
    zs = z - 0.35 * (8 - 1) - 1.7
    iz = np.int(np.fix(z) - 12)
    u = XU[iz]
    el3 = (IE3[iz])
    el1 = (IE1[iz])

    q = qa02 / (zs**2)
    kh2 = (E / (r * zs**2)) - 0.25
    akh = np.sqrt(abs(kh2))
    if kh2 >= 0:
        d = 1 - np.exp(-2 * np.pi / akh)
        bp = np.arctan(akh / (q - kh2 + 0.25))
        if bp < 0:
            bp = bp + np.pi
        c = np.exp((-2 / akh) * bp)
    else:
        d = 1
        c = np.exp((-1 / akh)
                   * np.log((q + 0.25 - kh2 + akh)
                            / (q + 0.25 - kh2 - akh)))

    if E - el1 <= 0:
        g = 2.25 * q**4 - (0.75 + 3 * kh2) * q**3 \
            + (0.59375 - 0.75 * kh2 - 0.5 * kh2**2) * q * q \
            + (0.11146 + 0.85417 * kh2 + 1.8833 * kh2 * kh2 + kh2**3) * q \
            + 0.0035807 + kh2 / 21.333 + kh2 * kh2 / 4.5714 \
            + kh2 ** 3 / 2.4 + kh2**4 / 4

        a = ((q - kh2 + 0.25)**2 + kh2)**5
    else:
        g = q**3 - (5 / 3 * kh2 + 11 / 12) * q ** 2 \
            + (kh2 * kh2 / 3 + 1.5 * kh2 + 65 / 48) * q \
            + kh2**3 / 3 + 0.75 * kh2 * kh2 \
            + 23 / 48 * kh2 + 5 / 64

        a = ((q - kh2 + 0.25)**2 + kh2)**4

    rf = ((E + 0.1 - el3) / 1.8 / z / z)**u
    if(np.abs(iz - 11) <= 5 and E - el3 <= 20):
        rf = 1

    out = rf * 32 * g * c / a / d * E / r / r / zs**4
    return out


def sigpar(z, dl, shell, e0, beta):
    '''
    Calculate parameterized partial cross section of major ionization edge

    Python implementation of Matlab code from R. F. Egerton at:
    http://www.tem-eels.ca/

    Based on relativistic oscillator strength and kinematics for an isotropic
    material. Valid only for a limited collection angle, falling within the
    dipole region (beta << sqrt(E/E0)).  Relies on optical measures of
    oscillator strength in the files FK.DAT, FL.DAT, FM23.DAT, FM45.DAT, and
    FM045.DAT. Based on values given in Ultramicroscpy 50 (1993) 13-28.
    Details in R.F.Egerton: EELS in the Electron Microscope, 3rd edition,
    Springer 2011

    Args
    ----
    z : int
        Atomic number of scattering atom
    dl : int
        Integration window for calculation (eV)  Should be within the range
        30 to 250 eV
    shell : str
        Must be: 'K', 'L', 'M23', 'M45', 'N', or 'O'
    e0 : int or float
        Electron energy (keV)
    beta : int or float
        Maximum scatterting semi-angle for integration (mrads). If it exceeds
        half the Bethe ridge angle (outside the dipole region), a warning is
        given.

    '''

    def fdcalc(dl, f50, f100, f200):
        '''
        Calculate f(delta) values
        '''
        if dl <= 50:
            fd = f50 * dl / 50

        elif((dl > 50) and (dl < 100)):
            fd = f50 + (dl - 50) / 50 * (f100 - f50)

        elif((dl >= 100) and (dl < 250)):
            fd = f100 + (dl - 100) / 100 * (f200 - f100)
        return fd

    shell = shell.upper()

    print('---------------Sigpar----------------')
    print('Z: %g' % z)
    print('Delta (eV): %g' % dl)
    print('Edge type: %s' % shell)

    # Select f-values table based on edge type
    if shell == 'K':
        infile = 'Sigpar_fk.dat'
    elif shell == 'L':
        infile = 'Sigpar_fl.dat'
    elif shell == 'M23':
        infile = 'Sigpar_fm23.dat'
    elif shell == 'M45':
        infile = 'Sigpar_fm45.dat'
    elif (shell == 'N') or (shell == 'O'):
        infile = 'Sigpar_fno45.dat'
    else:
        raise ValueError("Invalid Edge Type ''%s''" % shell)
    infile = imp.find_module("emtools")[1] + "/data/" + infile

    # Read edge type data
    inData = np.loadtxt(infile)

    # Lookup z value in edge type table
    idx = np.where(inData[:, 0] == z)
    fdata = inData[idx, :][0][0]

    # Get f-values from table
    if (shell == 'K') or (shell == 'L') or (shell == 'M45'):
        ec = fdata[1]
        f50 = fdata[2]
        f100 = fdata[3]
        f200 = fdata[4]
        erp = fdata[5]
        fd = fdcalc(dl, f50, f100, f200)
    elif shell == 'M23':
        ec = fdata[1]
        f30 = fdata[2]
        dl = 30
        fd = f30
        erp = 10
        print('For delta = 30eV\n')
    elif (shell == 'N') or (shell == 'O'):
        ec = fdata[1]
        f50 = fdata[2]
        f100 = fdata[3]
        erp = fdata[4]
        fd = fdcalc(dl, f50, f100, f100)

    # Get e0 and beta
    print('Ec = %0.15g eV,  f(delta) =  %0.15g ' % (ec, fd))
    print('E0 (keV): %g' % e0)
    print('beta(mrad): %g' % beta)

    if (beta ^ 2) > (50 * ec / e0):
        print('Dipole Approximation NOT VALID, sigma will be too high!\n')

    # Calculate Sigma
    ebar = np.sqrt(ec * (ec + dl))
    gamma = 1 + e0 / 511
    g2 = gamma ** 2
    v2 = 1 - 1 / g2
    b2 = beta ** 2
    thebar = ebar / e0 / (1 + 1 / gamma)
    t2 = thebar * thebar
    gfunc = np.log(g2) - np.log((b2 + t2) / (b2 + t2 / g2))\
        - v2 * b2 / (b2 + t2 / g2)
    squab = np.log(1 + b2 / t2) + gfunc
    sigma = 1.3e-16 * g2 / (1 + gamma) / ebar / e0 * fd * squab
    print('sigma = %0.3g cm^2' % sigma)
    if np.logical_not(((beta**2) > (50 * ec / e0))):
        print('Estimated accuracy = %0.4g %%' % erp)

def calc_Z_eff(composition, elements):
    Zs = np.array([hs.material.elements[i].General_properties.Z for i in elements])
    Z_eff = np.sum(composition * Zs**1.3)/np.sum(composition * Zs**0.3)
    return Z_eff

def calc_mean_energy_loss(Z_eff):
    return 7.6 * Z_eff**0.36

def calc_mean_free_path(spec, method='iak', density=None, composition=None, elements=None):
    """
    Estimate the mean free path for inelastic scattering.
    
    'iak' method uses Hyperspy's iMFP_Iakoubovskii function an is based on equation 9 in: 
        Iakoubovskii, K., K. Mitsuishi, Y. Nakayama, and K. Furuya.
        ‘Thickness Measurements with Electron Energy Loss Spectroscopy’.
        Microscopy Research and Technique 71, no. 8 (2008): 626–31.
        https://doi.org/10.1002/jemt.20597

    'malis' method is based on equations 6, 7, and 8 in:
        T. Malis et al., J. Electron Microsc. Tech. vol. 8 (1988) 193.
    
    Collection angle correction adapted from CONCOR2.  Details are in:
        R.F.Egerton: EELS in the Electron Microscope, 3rd edition, Springer 2011

    Args
    ----------
    spec : Hyperspy EELSSpectrum
        Low-loss EELS spectrum
    method : string
        Either 'iak' or 'malis'
    density : float
        Specimen density in g/cm**3. Required for 'iak' method.
    composition : list or NumPy array
        Composition of specimen in at.%. Required for 'malis' method.
    elements : list
        List of strings giving the element for each composition. Required for 'malis' method.

    Returns
    ----------
    mean_free_path : float
        Estimated mean free path.
    """
    beam_energy = spec.metadata.Acquisition_instrument.TEM.beam_energy
    alpha = spec.metadata.Acquisition_instrument.TEM.convergence_angle
    beta = spec.metadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle

    theta_C = 20 #mrads
    F = (1 + (beam_energy / 1022)) / (1 + (beam_energy / 511))**2
    theta_E = 5.5 * density ** 0.3 / (F * beam_energy)

    if method.lower() == 'iak':
        if density is None:
            raise ValueError('For Iakoubovskii method density must be provided')
        else:
            numerator = (alpha ** 2 + beta ** 2 + 2 * theta_E**2 + abs(alpha ** 2 - beta ** 2)) * theta_C**2
            denominator = (alpha ** 2 + beta ** 2 + 2 * theta_C ** 2 + abs(alpha ** 2 - beta ** 2)) * theta_E**2
            inv_lambda = (11*density**0.3/(200*F*beam_energy)) * np.log(numerator/denominator)
            mean_free_path = 1/inv_lambda

    elif method.lower() == 'malis':
        if composition is None or elements is None or beam_energy is None:
            raise ValueError('For Malis method, the composition and elements must be provided')
        if len(composition) != len(elements):
            raise ValueError('Number of compositions inconsistent with number of elements')
        Z_eff = calc_Z_eff(composition, elements)
        Em = calc_mean_energy_loss(Z_eff)

        eta1 = np.sqrt((alpha**2 + beta**2 + theta_E**2)**2 - 4. * alpha**2 * beta**2) - alpha**2 - beta**2 - theta_E**2
        eta2 = 2.*beta**2*np.log(0.5/theta_E**2*(np.sqrt((alpha**2+theta_E**2-beta**2)**2 + 4.*beta**2*theta_E**2)+alpha**2+theta_E**2-beta**2))
        eta3 = 2.*alpha**2*np.log(0.5/theta_E**2*(np.sqrt((beta**2+theta_E**2-alpha**2)**2 + 4.*alpha**2*theta_E**2)+beta**2+theta_E**2-alpha**2))

        eta=(eta1+eta2+eta3)/alpha**2/np.log(4./theta_E**2)
        f1=(eta1+eta2+eta3)/2./alpha**2/np.log(1.+beta**2/theta_E**2)
        f2=f1
        if alpha>beta:
            f2 = f1*alpha**2/b2
        beta_eff = theta_E * np.sqrt(np.exp(f2 * np.log(1.+beta**2/theta_E**2))-1.)
        beta = beta_eff

        mean_free_path = 106 * F * beam_energy / (Em * np.log(2 * beta_eff * beam_energy / Em))
    return mean_free_path