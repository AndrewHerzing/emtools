from scipy.io import savemat
import numpy as np


def save_axsia(s, filename=None):
    if not filename:
        filename = 'matlab.mat'
    savedict = {}
    savedict['nrows'] = np.float64(s.data.shape[1])
    savedict['ncols'] = np.float64(s.data.shape[0])
    savedict['nchannels'] = np.float64(s.data.shape[2])
    savedict['resolution'] = np.float64(s.axes_manager[-1].scale)
    s.unfold()
    savedict['specdata'] = s.data.T

    savemat(filename, savedict, format='5')

    s.fold()

    return
