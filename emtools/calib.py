# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
Calibrations module for EMTools package.

@author: Andrew Herzing
"""


def get_lattice_spacings(material):
    """
    Return primary lattice spacings for a given material.

    Args
    ----------
    material : string
        Must be one of either 'AuPd', 'Au', or 'Si'

    Returns
    ----------
    hkls : Dictionary
        Primary lattice spacings in Angstroms

    """
    aupd = {'111': 2.31, '200': 2.00, '220': 1.41, '311': 1.21, '222': 1.15}
    au = {'111': 2.36, '200': 2.04, '220': 1.44, '311': 1.23, '222': 1.18}
    si = {'111': 3.14, '200': 2.72, '220': 1.92, '311': 1.64, '222': 1.57}

    if material.lower() == 'aupd':
        hkls = aupd
    elif material.lower() == 'au':
        hkls = au
    elif material.lower() == 'si':
        hkls = si
    else:
        raise ValueError("Unknown material.  Must be 'Au-Pd', 'Au', or 'Si'")
    return hkls
