import numpy as np


def calc_hkl(hkl, unit_cell='cubic', a=None, b=None, c=None,
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
    a : float
        Lattice parameter a of crystal.
    b : float
        Lattice parameter b of crystal.
    c : float
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
    H = hkl[0]
    K = hkl[1]
    L = hkl[2]

    if alpha:
        alpha = alpha * np.pi / 180
    if beta:
        beta = beta * np.pi / 180
    if gamma:
        gamma = gamma * np.pi / 180

    if unit_cell == 'cubic':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        d = np.sqrt(a**2 / (H**2 + K**2 + L**2))

    elif unit_cell == 'tetragonal':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not c:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        d = np.sqrt(1 / (((H**2 + K**2) / a**2) + (L**2 / c**2)))

    elif unit_cell == 'hexagonal':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not c:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        d = np.sqrt(1 / (4 / 3 * ((H**2 + H * K + K**2) / a**2) +
                    (L**2 / c**2)))

    elif unit_cell == 'rhombohedral' or unit_cell == 'trigonal':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not alpha:
            raise ValueError("Must provide lattice constant 'alpha' for unit\
                cell type '%s'." % unit_cell)
        d = np.sqrt((1 / (((H**2 + H * K + K**2) * (np.sin(alpha))**2
                    + 2 * (H * K + K * L + L * H) * ((np.cos(alpha))**2 -
                    np.cos(alpha))) / (a**2 * (1 + 2 * (np.cos(alpha))**3 -
                                               3 * (np.cos(alpha))**2)))))

    elif unit_cell == 'orthorhombic':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not b:
            raise ValueError("Must provide lattice constant 'b' for unit cell\
                type '%s'." % unit_cell)
        if not c:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        d = np.sqrt(1 / ((H**2 / a**2) + (K**2 / b**2) + (L**2 / c**2)))

    elif unit_cell == 'monoclinic':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not b:
            raise ValueError("Must provide lattice constant 'b' for unit cell\
                type '%s'." % unit_cell)
        if not c:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        if not beta:
            raise ValueError("Must provide lattice constant 'beta' for unit\
                cell type '%s'." % unit_cell)
        d = np.sqrt(1 / ((1 / (np.sin(beta))**2) * ((H**2 / a**2) +
                    ((K**2 * (np.sin(beta))**2) / b**2) + (L**2 / c**2) -
                    (2 * H * L * np.cos(beta) / (a * c)))))

    elif unit_cell == 'triclinic':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not b:
            raise ValueError("Must provide lattice constant 'b' for unit cell\
                type '%s'." % unit_cell)
        if not c:
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
        S11 = b**2 * c**2 * (np.sin(alpha))**2
        S22 = a**2 * c**2 * (np.sin(beta))**2
        S33 = a**2 * b**2 * (np.sin(gamma))**2
        S12 = a * b * c**2 * (np.cos(alpha) * np.cos(beta) - np.cos(gamma))
        S23 = a**2 * b * c * (np.cos(beta) * np.cos(gamma) - np.cos(alpha))
        S13 = a * b**2 * c * (np.cos(gamma) * np.cos(alpha) - np.cos(beta))
        V = a * b * c * np.sqrt(1 - (np.cos(alpha))**2 - (np.cos(beta))**2 -
                                (np.cos(gamma))**2 + 2 *
                                np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        d = np.sqrt(1 / ((1 / V**2) * (S11 * H**2 + S22 * K**2 + S33 * L**2 +
                         2 * S12 * H * K + 2 * S23 * K * L + 2 * S13 * H * L)))
    else:
        raise ValueError("%s is not a valid unit cell type. Must be 'cubic',"
                         "'tetragonal', 'hexagonal', 'rhombohedral',"
                         "'trigonal', 'orthorhomic', 'monoclinic',"
                         "or 'triclinic'" % unit_cell)
    return np.around(d, decimals)
