from hyperspy.misc import elements
from matplotlib import pylab as plt


def plotEDS(spec, axis=None, peaklabels=None, linecolor='red',
            energyrange=None, intensityrange=None, horz_offset=None,
            vert_offset=None, fontsize=8):
    if axis is None:
        figure, axis = plt.subplots(1)
        out = True
    else:
        out = False
    axis.plot(spec.axes_manager[-1].axis, spec.data, color=linecolor)
    if energyrange:
        axis.set_xlim(energyrange[0], energyrange[1])
    if intensityrange:
        axis.set_ylim(intensityrange[0], intensityrange[1])
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
                          size=fontsize)
    if out:
        return(figure, axis)
    else:
        return
