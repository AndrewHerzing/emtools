# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
Diffraction module for EMTools package

@author: Andrew Herzing
"""

import numpy as np


def calc_hkl(hkl, unit_cell='cubic', a_dim=None, b_dim=None, c_dim=None,
             alpha=None, beta=None, gamma=None, decimals=4):
    """
    Calculate d-spacing for hkl plane in a unit cell.

    Args
    ----------
    hkl : list
        Miller indices of the lattice plane
    unit_cell : str
        Unit cell of crystal. Must be 'cubic', 'tetragonal', 'hexagonal',
        'rhombohedral', 'trigonal', 'orthorhomic', 'monoclinic', or 'triclinic'
    a_dim : float
        Lattice parameter a of crystal.
    b_dim : float
        Lattice parameter b of crystal.
    c_dim : float
        Lattice parameter c of crystal.
    alpha : float
        Lattice angle alpha of crystal.
    beta : float
        Lattice angle beta of crystal.
    gamma : float
        Lattice angle gamma of crystal.
    decimals : int
        Number of decimal points to include in calculated d-spacing. Default
        is 4.

    Returns
    ----------
    d : float
        Calculated spacing for the lattice plane and unit cell provided

    """

    if len(hkl) != 3:
        raise ValueError("'hkl' is not valid.")
    h_index = hkl[0]
    k_index = hkl[1]
    l_index = hkl[2]

    if alpha:
        alpha = alpha * np.pi / 180
    if beta:
        beta = beta * np.pi / 180
    if gamma:
        gamma = gamma * np.pi / 180

    if unit_cell == 'cubic':
        if not a_dim:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        d_hkl = np.sqrt(a_dim**2 / (h_index**2 + k_index**2 + l_index**2))

    elif unit_cell == 'tetragonal':
        if not a_dim:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not c_dim:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        d_hkl = np.sqrt(1 / (((h_index**2 + k_index**2) / a_dim**2)
                        + (l_index**2 / c_dim**2)))

    elif unit_cell == 'hexagonal':
        if not a_dim:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not c_dim:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        d_hkl = np.sqrt(1 / (4 / 3 *
                        ((h_index**2 + h_index * k_index + k_index**2)
                         / a_dim**2) + (l_index**2 / c_dim**2)))

    elif unit_cell == 'rhombohedral' or unit_cell == 'trigonal':
        if not a_dim:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not alpha:
            raise ValueError("Must provide lattice constant 'alpha' for unit\
                cell type '%s'." % unit_cell)
        d_hkl = np.sqrt((1 / (((h_index**2 + h_index * k_index + k_index**2)
                               * (np.sin(alpha))**2
                               + 2 * (h_index * k_index + k_index
                                      * l_index + l_index * h_index)
                               * ((np.cos(alpha))**2 - np.cos(alpha)))
                              / (a_dim**2 * (1 + 2 * (np.cos(alpha))**3
                                 - 3 * (np.cos(alpha))**2)))))

    elif unit_cell == 'orthorhombic':
        if not a_dim:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not b_dim:
            raise ValueError("Must provide lattice constant 'b' for unit cell\
                type '%s'." % unit_cell)
        if not c_dim:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        d_hkl = np.sqrt(1 / ((h_index**2 / a_dim**2) + (k_index**2 / b_dim**2)
                        + (l_index**2 / c_dim**2)))

    elif unit_cell == 'monoclinic':
        if not a_dim:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not b_dim:
            raise ValueError("Must provide lattice constant 'b' for unit cell\
                type '%s'." % unit_cell)
        if not c_dim:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        if not beta:
            raise ValueError("Must provide lattice constant 'beta' for unit\
                cell type '%s'." % unit_cell)
        d_hkl = np.sqrt(1 / ((1 / (np.sin(beta))**2) * ((h_index**2 / a_dim**2)
                        + ((k_index**2 * (np.sin(beta))**2) / b_dim**2)
                        + (l_index**2 / c_dim**2)
                        - (2 * h_index * l_index * np.cos(beta)
                           / (a_dim * c_dim)))))

    elif unit_cell == 'triclinic':
        if not a_dim:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not b_dim:
            raise ValueError("Must provide lattice constant 'b' for unit cell\
                type '%s'." % unit_cell)
        if not c_dim:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        if not alpha:
            raise ValueError("Must provide lattice constant 'alpha' for unit\
                cell type '%s'." % unit_cell)
        if not beta:
            raise ValueError("Must provide lattice constant 'beta' for unit\
                cell type '%s'." % unit_cell)
        if not gamma:
            raise ValueError("Must provide lattice constant 'gamma' for unit\
                cell type '%s'." % unit_cell)
        s11 = b_dim**2 * c_dim**2 * (np.sin(alpha))**2
        s22 = a_dim**2 * c_dim**2 * (np.sin(beta))**2
        s33 = a_dim**2 * b_dim**2 * (np.sin(gamma))**2
        s12 = a_dim * b_dim * c_dim**2\
            * (np.cos(alpha) * np.cos(beta) - np.cos(gamma))
        s23 = a_dim**2 * b_dim * c_dim\
            * (np.cos(beta) * np.cos(gamma) - np.cos(alpha))
        s13 = a_dim * b_dim**2 * c_dim\
            * (np.cos(gamma) * np.cos(alpha) - np.cos(beta))
        vol = a_dim * b_dim * c_dim\
            * np.sqrt(1 - (np.cos(alpha))**2 - (np.cos(beta))**2
                      - (np.cos(gamma))**2 + 2 * np.cos(alpha) * np.cos(beta)
                      * np.cos(gamma))
        d_hkl = np.sqrt(1 / ((1 / vol**2)
                        * (s11 * h_index**2 + s22 * k_index**2
                           + s33 * l_index**2
                        + 2 * s12 * h_index * k_index
                        + 2 * s23 * k_index * l_index
                        + 2 * s13 * h_index * l_index)))
    else:
        raise ValueError("%s is not a valid unit cell type. Must be 'cubic',"
                         "'tetragonal', 'hexagonal', 'rhombohedral',"
                         "'trigonal', 'orthorhomic', 'monoclinic',"
                         "or 'triclinic'" % unit_cell)
    return np.around(d_hkl, decimals)
