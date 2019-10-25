# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
EDS module for EMTools package

@author: Andrew Herzing
"""

from hyperspy.misc import elements
from matplotlib import pylab as plt


def plot_EDS(spec, axis=None, peaklabels=None, line_color='red',
             energy_range=None, intensity_range=None, horz_offset=None,
             vert_offset=None, font_size=8):
    """
    Plot several EDS spectra.

    Args
    ----------
    spec : Hyperspy Signal1D
        Single EDS spectrum signal
    axis : Matplotlib axis
        Axis in which to plot the data.  If None, a new Figure and Axis are
        created
    peak_labels : bool or list
        If True, label the peaks defined in spec.metadata.Sample.xray_lines.
        If list, the listed peaks are labeled.
    line_color : string
        Color for the spectral plots
    energy_range : tuple
        Plot is truncated horizonatally to the minimum and maximum value
    intensity_range : tuple
        Plot is truncated vertically to the minimum and maximum value
    horz_offset : float
        Offset from peak location (in calibrated values) with which to offset
        the labels in the horizontal direction
    vert_offset : float
        Offset from peak location (in calibrated values) with which to offset
        the labels in the vertical direction
    font_size : int
        Fontsize for labels


    Returns
    ----------
    figure : Matplotlib Figure instance
    axis : Matplotlib Axis instance

    """
    if axis is None:
        figure, axis = plt.subplots(1)
        out = True
    else:
        out = False
    axis.plot(spec.axes_manager[-1].axis, spec.data, color=line_color)
    if energy_range:
        axis.set_xlim(energy_range[0], energy_range[1])
    if intensity_range:
        axis.set_ylim(intensity_range[0], intensity_range[1])
    if peaklabels:
        if peaklabels is True:
            peaklabels = spec.metadata.Sample.xray_lines
        elif type(peaklabels) is list:
            pass
        else:
            raise ValueError("Unknown format for 'peaklabels'. "
                             "Must be boolean or list")
            return
        if vert_offset is None:
            vert_min, vert_max = axis.get_ylim()
            vert_offset = 0.05 * vert_max
        if horz_offset is None:
            horz_min, horz_max = axis.get_xlim()
            horz_offset = 0.01 * horz_max
        for i in range(0, len(peaklabels)):
            element, line = peaklabels[i].split('_')
            energy = (elements.elements[element]
                      ['Atomic_properties']
                      ['Xray_lines']
                      [line]
                      ['energy (keV)'])
            y_pos = spec.isig[energy].data + vert_offset
            x_pos = energy + horz_offset
            if y_pos > vert_max:
                y_pos = vert_max + 0.01 * vert_max
            if (x_pos < horz_min) or (x_pos > horz_max):
                pass
            else:
                axis.text(x=x_pos,
                          y=y_pos,
                          s=peaklabels[i],
                          rotation=90,
                          rotation_mode='anchor',
                          size=font_size)
    if out:
        return figure, axis
    else:
        return
