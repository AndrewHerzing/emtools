import numpy as np
import hyperspy.api as hs
from pystackreg import StackReg


def get_stackreg_shifts(stack):
    sr = StackReg(StackReg.TRANSLATION)
    sr_shifts = sr.register_stack(stack.data)
    sr_shifts = np.array([sr_shifts[i][:-1, 2][::-1]
                          for i in range(0, len(sr_shifts))])
    return sr_shifts


def apply_hanning(image):
    h = np.hanning(image.data.shape[0])
    ham2d = np.sqrt(np.outer(h, h))
    image.data = image.data * ham2d
    return image


def apply_taper(image, taper_percent):
    width = np.int32(np.round(taper_percent / 100 * image.data.shape[0]))
    image.data = np.pad(image.data, pad_width=width, mode='linear_ramp')
    return image


def get_ps(s, crop=True, hanning=False, taper=False, taper_percent=3,
           crop_factor=4):
    image = s.deepcopy()
    if hanning:
        image = apply_hanning(image)
    if taper:
        image = apply_taper(image, taper_percent)

    ps = hs.signals.Signal2D(np.log(image.fft(shift=True).amplitude()))
    if crop:
        offset = ps.data.shape[0] / crop_factor
        center = ps.data.shape[0] / 2
        ps = ps.isig[center - offset:center + offset,
                     center - offset:center + offset]
    return ps
