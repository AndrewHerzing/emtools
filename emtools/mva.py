# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
MVA module for EMTools package.

@author: Andrew Herzing
"""

import numpy as np
import tqdm
from scipy import optimize
from pymcr.mcr import McrAR


def varimax(factors, gamma=1.0, maxiters=250, tol=1e-5):
    """
    Perform Varimax rotation on a data array.

    The rotation is calculated iteratively by attempting to maximize the sum
    of the variances of the squared loadings of each factor.

    Args
    ----------
    factors : NumPy array
        Array of factors to be rotated (p x k)
    gamma : fload
        Degree of rotation.  Default is 1.0 which cooresponds to the Varimax rotation.
    maxiters : int
        Maximum number of iterations for calculating the rotation
    tol : float
        Tolerance for convergence

    Returns
    -----------
    rotated : Numpy array
        Rotated version of P with shape (p x k)
    R : Numpy array
        Rotation matrix with shape (k x k)

    """
    p, k = factors.shape
    R = np.eye(k)

    var_old = 0
    iter = 0
    for i in range(maxiters):
        Lambda = factors @ R
        u, s, v = np.linalg.svd((factors.T @ (np.power(Lambda, 3) - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.power(Lambda, 2)))))))
        R = u @ v
        var = np.sum(s)
        if (iter > 0) and (np.abs((var - var_old) / var) < tol):
            break
        iter += 1
    if iter == maxiters:
        print('Rotation did not converge after maximum number of iterations (%s)' % maxiters)
    rotated = factors @ R
    return rotated, R


def weight_si(si):
    """
    Normalize a spectrum image data array to account for Poissonian statistics.

    Args
    ----------
    si : NumPy array
        Array of size (nchannels, npixels) or (nchannels, nrows, ncols)

    Returns
    -----------
    normalized : Numpy array
        Normalized version of input array of size (nchannels, npixels)
    wt_spec : Numpy array
        Weighting vector for the spectral dimension
    wt_im : Numpy array
        Weighting vector for the spatial dimension

    """
    if len(si.shape) == 2:
        pass
    elif len(si.shape) == 3:
        nchannels, nrows, ncols = si.shape
        npixels = nrows * ncols
        si = si.reshape([nchannels, npixels])
    si += 1e-6
    wt_spec = 1 / np.sqrt(si.mean(0))
    wt_spec[~np.isfinite(wt_spec)] = 0
    wt_im = 1 / np.sqrt(si.mean(1))
    wt_im[~np.isfinite(wt_im)] = 0

    normalized = si * wt_im[:, np.newaxis] * wt_spec[np.newaxis, :]
    return normalized, wt_spec, wt_im


def flip_to_positive(loadings, factors):
    """
    Flip factors to positive

    Args
    ----------
    loadings : NumPy array
        Loadings array of shape (npixels, nfactors)
    factors : fload
        Factor array of shape (nchannels, nfactors)

    Returns
    -----------
    loadings : Numpy array
        Modified versin of the loadings array where the components
        have been flipped to positive if neccessary.
    factors : Numpy array
        Modified version of the factors array where the components
        have been flipped to positive if neccessary.

    """
    _, nloadings = loadings.shape
    _, nfactors = factors.shape
    if nloadings != nfactors:
        raise ValueError('Number of columns must be the same in both the loadings and the factors array')
    for i in range(nloadings):
        if (factors[:, i].sum() < 0) and (loadings[:, i].sum() < 0):
            loadings[:, i] *= -1
            factors[:, i] *= -1
    return loadings, factors


def spatial_simplicity(A, D, wt_spec, wt_im, sum_to_one=True):
    mcrar = McrAR()
    mcrar.fit(D, C=A, verbose=True)
    factors_SS = mcrar.ST_opt_
    factors_SS = (factors_SS / wt_spec)

    ST = mcrar.ST_opt_
    mcrar = McrAR(max_iter=1, tol_increase=0.)
    mcrar.fit(D, ST=ST, verbose=True)
    loadings_SS = mcrar.C_opt_
    loadings_SS = (loadings_SS.T / wt_im).T
    if sum_to_one:
        loadings_SS = (loadings_SS / loadings_SS.sum(0))
        factors_SS = (factors_SS.T / factors_SS.sum(1)).T
    return loadings_SS, factors_SS


def upsample_factors(si, loadings, factors, scaling, **kwargs):
    """
    Upsample calculated factors to the original resolution

    Args
    ----------
    si : Hyperspy Signal1D
        Original hyperspectral dataset before decomposition
    loadings : NumPy array
        Loadings array of shape (npixels, nfactors)
    factors : fload
        Factor array of shape (nfactors, nchannels)
    scaling : list or NumPy array
        Factors by which the original SI was compressed in order to perform
        the decomposition.  The order must be:
        [compression_rows, compression_columns, compression_spectral]

    Returns
    -----------
    C_nnls : Numpy array
        Upsampled loadings array [nfactors, nrows, ncols]
    ST_nnls : Numpy array
        Upsampled spectral factors array [nchannels, nfactors]

    """
    tol = kwargs.get('tol', None)
    maxiter = kwargs.get('maxiter', None)
    ST = factors.T
    C = loadings
    nfactors = C.shape[1]
    nrows, ncols, nchannels = si.data.shape
    npixels = nrows * ncols
    si_binned = si.rebin(scale=(1, 1, scaling[2]))
    si_binned.unfold()
    si_binned = si_binned.data + 1e-6
    C_nnls = np.zeros([nfactors, si_binned.shape[0]])
    for i in tqdm.tqdm(range(npixels)):
        C_nnls[:, i], _ = optimize.nnls(ST, si_binned[i, :], maxiter=maxiter, atol=tol)

    si_binned = si.rebin(scale=(scaling[0], scaling[1], 1))
    si_binned.unfold()
    si_binned = si_binned.data + 1e-6
    ST_nnls = np.zeros([nfactors, si_binned.shape[1]])
    for i in tqdm.tqdm(range(nchannels)):
        ST_nnls[:, i], _ = optimize.nnls(C, si_binned[:, i], maxiter=maxiter, atol=tol)
    C_nnls = C_nnls.reshape([nfactors, nrows, ncols])
    return C_nnls, ST_nnls.T
