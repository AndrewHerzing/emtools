import numpy as np


def calc_hkl(hkl, unit_cell='cubic', a=None, b=None, c=None,
             alpha=None, beta=None, gamma=None, decimals=4):

    if len(hkl) != 3:
        raise ValueError("'hkl' is not valid.")
    h = hkl[0]
    k = hkl[1]
    l = hkl[2]

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
        d = np.sqrt(a**2 / (h**2 + k**2 + l**2))

    elif unit_cell == 'tetragonal':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not c:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        d = np.sqrt(1 / (((h**2 + k**2) / a**2) + (l**2 / c**2)))

    elif unit_cell == 'hexagonal':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not c:
            raise ValueError("Must provide lattice constant 'c' for unit cell\
                type '%s'." % unit_cell)
        d = np.sqrt(1 / (4 / 3 * ((h**2 + h * k + k**2) / a**2) +
                    (l**2 / c**2)))

    elif unit_cell == 'rhombohedral' or unit_cell == 'trigonal':
        if not a:
            raise ValueError("Must provide lattice constant 'a' for unit cell\
                type '%s'." % unit_cell)
        if not alpha:
            raise ValueError("Must provide lattice constant 'alpha' for unit\
                cell type '%s'." % unit_cell)
        d = np.sqrt((1 / (((h**2 + h * k + k**2) * (np.sin(alpha))**2
                    + 2 * (h * k + k * l + l * h) * ((np.cos(alpha))**2 -
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
        d = np.sqrt(1 / ((h**2 / a**2) + (k**2 / b**2) + (l**2 / c**2)))

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
        d = np.sqrt(1 / ((1 / (np.sin(beta))**2) * ((h**2 / a**2) +
                    ((k**2 * (np.sin(beta))**2) / b**2) + (l**2 / c**2) -
                    (2 * h * l * np.cos(beta) / (a * c)))))

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
        d = np.sqrt(1 / ((1 / V**2) * (S11 * h**2 + S22 * k**2 + S33 * l**2 +
                         2 * S12 * h * k + 2 * S23 * k * l + 2 * S13 * h * l)))
    else:
        raise ValueError("%s is not a valid unit cell type. Must be 'cubic',"
                         "'tetragonal', 'hexagonal', 'rhombohedral',"
                         "'trigonal', 'orthorhomic', 'monoclinic',"
                         "or 'triclinic'" % unit_cell)
    return np.around(d, decimals)

    return d
