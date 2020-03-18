import numpy as np
import cv2
from scipy import ndimage
import tqdm


def align_stack(stack, method, start, show_progressbar):
    """
    Compute the shifts for spatial registration.

    Shifts are determined by one of three methods:
        1.) Phase correlation (PC) as implemented in OpenCV.
        2.) Enhanced correlation coefficient (ECC) as implemented in OpenCV.

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

    def calculate_shifts(stack, method, start, show_progressbar, nslice):
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

    shifts = calculate_shifts(stack, method, start, show_progressbar)
    composed = compose_shifts(shifts, start)
    aligned = apply_shifts(stack, composed)
    return aligned
