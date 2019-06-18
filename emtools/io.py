from scipy.io import savemat, loadmat
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


def axsia_to_hspy(filename, calibration_signal=None, im_shape=None):
    axsia = {}
    
    axsia_in = loadmat(filename)
    if 'nrows' and 'ncols' in axsia_in.keys():
        nrows = axsia_in['nrows'][0][0]
        ncols = axsia_in['ncols'][0][0]
    elif calibration_signal:
        nrows = calibration_signal.data.shape[0]
        ncols = calibration_signal.data.shape[1]
    elif im_shape:
        nrows = im_shape[0]
        ncols = im_shape[1]
    else:
        raise ValueError('SVD Decomposition requires definition of image shape')
    
    if 'npures' in axsia_in.keys():
        npures = axsia_in['npures'][0][0]
    else:
        npures = axsia_in['purespectra'].shape[1]
    if 'nchannels' in axsia_in.keys():
        nchannels = axsia_in['nchannels'][0][0]
    else:
        nchannels = axsia_in['purespectra'].shape[0]
    loadings = axsia_in['concentrations']
    factors = axsia_in['purespectra']
    method = axsia_in['SIparams'][0][0][6][0]
    
    axsia['loadings'] = hs.signals.Signal2D(loadings.reshape([nrows,ncols,npures])).swap_axes(0,1).swap_axes(1,2)
    axsia['factors'] = hs.signals.Signal1D(factors.reshape([nchannels,npures])).swap_axes(0,1)
    if calibration_signal:
        axsia['factors'].axes_manager[1].name = calibration_signal.axes_manager[-1].name
        axsia['factors'].axes_manager[1].offset = calibration_signal.axes_manager[-1].offset
        axsia['factors'].axes_manager[1].scale = calibration_signal.axes_manager[-1].scale
        axsia['factors'].axes_manager[1].units = calibration_signal.axes_manager[-1].units
        
        axsia['loadings'].axes_manager[1].name = calibration_signal.axes_manager[0].name
        axsia['loadings'].axes_manager[1].offset = calibration_signal.axes_manager[0].offset
        axsia['loadings'].axes_manager[1].scale = calibration_signal.axes_manager[0].scale
        axsia['loadings'].axes_manager[1].units = calibration_signal.axes_manager[0].units
        
        axsia['loadings'].axes_manager[2].name = calibration_signal.axes_manager[1].name
        axsia['loadings'].axes_manager[2].offset = calibration_signal.axes_manager[1].offset
        axsia['loadings'].axes_manager[2].scale = calibration_signal.axes_manager[1].scale
        axsia['loadings'].axes_manager[2].units = calibration_signal.axes_manager[1].units

    return axsia
    
