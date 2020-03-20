import numpy as np
import cv2
from scipy import ndimage
import tqdm
import copy
import hyperspy.api as hs
from pystackreg import StackReg


def align_stack(stack, method, start, show_progressbar):
    """
    Compute the shifts for spatial registration.

    Shifts are determined by one of three methods:
        1.) Phase correlation (PC) as implemented in OpenCV.
        2.) Enhanced correlation coefficient (ECC) as implemented in OpenCV.
        3.) Intensity based method implemented in PyStackReg
            This method was first described in:
            P. Thevenaz, U.E. Ruttimann, M. Unser A Pyramid Approach to
            Subpixel Registration Based on Intensity. IEEE Transactions on
            Image Processing. vol. 7, no. 1, pp. 27-41, January 1998.

    Shifts are then applied and the aligned stack is returned.

    Args
    ----------
    stack : Numpy array
        3-D numpy array containing the tilt series data
    method : string
        Method by which to calculate the alignments. Valid options
        are 'PC' or 'ECC'.
    start : integer
        Position in tilt series to use as starting point for the alignment.
        If None, the central projection is used.
    show_progressbar : boolean
        Enable/disable progress bar

    Returns
    ----------
    out : TomoStack object
        Spatially registered copy of the input stack

    """
    def apply_shifts(stack, shifts):
        shifted = stack.deepcopy()
        for i in range(0, shifted.data.shape[0]):
            shifted.data[i, :, :] =\
                ndimage.shift(shifted.data[i, :, :],
                              shift=[shifts[i, 1], shifts[i, 0]],
                              order=0)
        if not shifted.original_metadata.has_item('shifts'):
            shifted.original_metadata.add_node('shifts')
        shifted.original_metadata.shifts = shifts
        return shifted

    def compose_shifts(shifts, start):
        if start is None:
            start = np.int32(np.floor((shifts.shape[0] + 1) / 2))
        composed = np.zeros([shifts.shape[0] + 1, 2])
        composed[start, :] = [0., 0.]
        for i in range(start + 1, composed.shape[0]):
            composed[i, :] = composed[i - 1, :] - shifts[i - 1, :]
        for i in range(start - 1, -1, -1):
            composed[i, :] = composed[i + 1, :] + shifts[i]
        return composed

    def calculate_shifts(stack, method, start, show_progressbar):
        shifts = np.zeros([stack.data.shape[0] - 1, 2])
        if start is None:
            start = np.int32(np.floor(stack.data.shape[0] / 2))

        if method == 'ECC':
            number_of_iterations = 1000
            termination_eps = 1e-3
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        number_of_iterations, termination_eps)

        for i in tqdm.tqdm(range(start, stack.data.shape[0] - 1),
                           disable=(not show_progressbar)):
            if method == 'PC':
                shifts[i, :] = cv2.phaseCorrelate(
                    np.float64(stack.data[i, :, :]),
                    np.float64(stack.data[i + 1, :, :]))[0]
            if method == 'ECC':
                if np.int32(cv2.__version__.split('.')[0]) == 4:
                    warp_matrix = np.eye(2, 3, dtype=np.float32)
                    (cc, trans) = cv2.findTransformECC(
                        np.float32(stack.data[i, :, :]),
                        np.float32(stack.data[i + 1, :, :]),
                        warp_matrix,
                        cv2.MOTION_TRANSLATION,
                        criteria,
                        inputMask=None,
                        gaussFiltSize=5)
                    shifts[i, :] = trans[:, 2]
                else:
                    warp_matrix = np.eye(2, 3, dtype=np.float32)
                    (cc, trans) = cv2.findTransformECC(
                        np.float32(stack.data[i, :, :]),
                        np.float32(stack.data[i + 1, :, :]),
                        warp_matrix,
                        cv2.MOTION_TRANSLATION,
                        criteria)
                    shifts[i, :] = trans[:, 2]

        if start != 0:
            for i in tqdm.tqdm(range(start - 1, -1, -1),
                               disable=(not show_progressbar)):
                if method == 'PC':
                    shifts[i, :] = cv2.phaseCorrelate(
                        np.float64(stack.data[i, :, :]),
                        np.float64(stack.data[i + 1, :, :]))[0]
                if method == 'ECC':
                    if np.int32(cv2.__version__.split('.')[0]) == 4:
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        (cc, trans) = cv2.findTransformECC(
                            np.float32(stack.data[i, :, :]),
                            np.float32(stack.data[i + 1, :, :]),
                            warp_matrix,
                            cv2.MOTION_TRANSLATION,
                            criteria,
                            inputMask=None,
                            gaussFiltSize=5)
                        shifts[i, :] = trans[:, 2]
                    else:
                        warp_matrix = np.eye(2, 3, dtype=np.float32)
                        (cc, trans) = cv2.findTransformECC(
                            np.float32(stack.data[i, :, :]),
                            np.float32(stack.data[i + 1, :, :]),
                            warp_matrix,
                            cv2.MOTION_TRANSLATION,
                            criteria)
                        shifts[i, :] = trans[:, 2]
        return shifts

    if method == 'StackReg':
        sr = StackReg(StackReg.TRANSLATION)
        tmats = sr.register_stack(stack.data, reference='previous')
        composed = np.zeros([len(tmats), 2])
        for i in range(0, len(tmats)):
            composed[i, :] = tmats[i][0:2, 2]
        aligned = sr.register_transform_stack(stack.data, reference='previous')
        aligned = hs.signals.Signal2D(aligned)
    else:
        shifts = calculate_shifts(stack, method, start, show_progressbar)
        composed = compose_shifts(shifts, start)
        aligned = apply_shifts(stack, composed)
    return aligned, composed


def align_to_other(other, shifts):
    """
    Spatially register a TomoStack using previously calculated shifts.

    Args
    ----------
    stack : TomoStack object
        TomoStack which was previously aligned
    other : TomoStack object
        TomoStack to be aligned. Must be the same size as the primary stack

    Returns
    ----------
    out : TomoStack object
        Aligned copy of other TomoStack

    """
    out = copy.deepcopy(other)

    for i in range(0, out.data.shape[0]):
        out.data[i, :, :] =\
            ndimage.shift(out.data[i, :, :],
                          shift=[shifts[i, 1], shifts[i, 0]],
                          order=0)
    return out


def get_ecc_error(stack, show_progressbar=False):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    cc = np.zeros(stack.data.shape[0] - 1)
    for i in tqdm.tqdm(range(0, stack.data.shape[0] - 1),
                       disable=(not show_progressbar)):
        (cc[i], trans) =\
            cv2.findTransformECC(np.float32(stack.data[i, :, :]),
                                 np.float32(stack.data[i + 1, :, :]),
                                 warp_matrix,
                                 cv2.MOTION_TRANSLATION)
    return cc


def apply_hanning(image):
    h = np.hanning(image.data.shape[0])
    ham2d = np.sqrt(np.outer(h, h))
    image.data = image.data * ham2d
    return image


def apply_taper(image, taper_percent):
    width = np.int32(np.round(taper_percent / 100 * image.data.shape[0]))
    image.data = np.pad(image.data, pad_width=width, mode='linear_ramp')
    return image


def get_ps(s, crop=True, hanning=False, taper=False, taper_percent=3):
    image = s.deepcopy()
    if hanning:
        image = apply_hanning(image)
    if taper:
        image = apply_taper(image, taper_percent)

    ps = hs.signals.Signal2D(np.log(image.fft(shift=True).amplitude()))
    if crop:
        offset = ps.data.shape[0] / 8
        center = ps.data.shape[0] / 2
        ps = ps.isig[center - offset:center + offset,
                     center - offset:center + offset]
    return ps
