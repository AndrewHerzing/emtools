import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

thermal = matplotlib.colors.\
    LinearSegmentedColormap.from_list('gatan_colormap',
                                      ['black', 'blue', 'green', 'red',
                                       'yellow', 'white'],
                                      256, 1.0)
JustRed = matplotlib.colors.\
    LinearSegmentedColormap.from_list('red_colormap',
                                      ['black', 'red'],
                                      256, 1.0)
JustGreen = matplotlib.colors.\
        LinearSegmentedColormap.from_list('green_colormap',
                                          ['black', 'green'],
                                          256, 1.0)
JustBlue = matplotlib.colors.\
    LinearSegmentedColormap.from_list('blue_colormap',
                                      ['black', 'blue'],
                                      256, 1.0)


def normalize(image):
    output = image - image.min()
    output = np.uint8(255*output/output.max())
    return(output)


def rgboverlay(im1, im2=None, im3=None):
    if len(np.shape(im1)) == 3:
        rgb = np.dstack((normalize(im1[:, :, 0]),
                         normalize(im1[:, :, 1]),
                         normalize(im1[:, :, 2])))
    else:
        rgb = np.dstack((normalize(im1),
                         normalize(im2),
                         normalize(im3)))
    return(rgb)


def genCMAP(color, alpha=None, N=256):
    if alpha:
        cmap = mpl.colors.\
            LinearSegmentedColormap.from_list('my_cmap', ['black', color], N)
        cmap._init()
        cmap._lut[:, -1] = np.linspace(0, alpha, cmap.N+3)
    else:
        cmap = mpl.colors.\
            LinearSegmentedColormap.from_list('my_cmap', ['black', color], N)
    return cmap


def mergeChannels(data, comps=None, colors=None, normalize='individual',
                  display=True):
    color_cycle = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta']
    if comps is None:
        if data.shape[2] <= 6:
            comps = np.arange(0, data.shape[2])
        else:
            comps = np.arange(0, 6)
    if colors is None:
        colors = color_cycle[0:len(comps)]
    images = {}
    images['rgb'] = np.zeros([data.shape[0], data.shape[1], 3])
    for i in colors:
        images[i] = np.zeros([data.shape[0], data.shape[1], 3])

    for i in range(0, len(comps)):
        if normalize == 'individual':
            if colors[i] == 'red':
                images['red'][:, :, 0] = \
                    data[:, :, comps[i]]/data[:, :, comps[i]].max()
            elif colors[i] == 'green':
                images['green'][:, :, 1] = \
                    data[:, :, comps[i]]/data[:, :, comps[i]].max()
            elif colors[i] == 'blue':
                images['blue'][:, :, 2] = \
                    data[:, :, comps[i]]/data[:, :, comps[i]].max()
            elif colors[i] == 'yellow':
                images['yellow'][:, :, 0] = \
                    data[:, :, comps[i]]/data[:, :, comps[i]].max()
                images['yellow'][:, :, 1] = \
                    data[:, :, comps[i]]/data[:, :, comps[i]].max()
            elif colors[i] == 'magenta':
                images['magenta'][:, :, 0] = \
                    data[:, :, comps[i]]/data[:, :, comps[i]].max()
                images['magenta'][:, :, 2] = \
                    data[:, :, comps[i]]/data[:, :, comps[i]].max()
            elif colors[i] == 'cyan':
                images['cyan'][:, :, 1] = \
                    data[:, :, comps[i]]/data[:, :, comps[i]].max()
                images['cyan'][:, :, 2] = \
                    data[:, :, comps[i]]/data[:, :, comps[i]].max()
            else:
                raise ValueError("Unknown color. Must be red, green, blue, "
                                 "yellow, magenta, or cyan")
        elif normalize == 'global':
            if colors[i] == 'red':
                images['red'][:, :, 0] = \
                    data[:, :, comps[i]]/data[:, :, comps].max()
            elif colors[i] == 'green':
                images['green'][:, :, 1] = \
                    data[:, :, comps[i]]/data[:, :, comps].max()
            elif colors[i] == 'blue':
                images['blue'][:, :, 2] = \
                    data[:, :, comps[i]]/data[:, :, comps].max()
            elif colors[i] == 'yellow':
                images['yellow'][:, :, 0] = \
                    data[:, :, comps[i]]/data[:, :, comps].max()
                images['yellow'][:, :, 1] = \
                    data[:, :, comps[i]]/data[:, :, comps].max()
            elif colors[i] == 'magenta':
                images['magenta'][:, :, 0] = \
                    data[:, :, comps[i]]/data[:, :, comps].max()
                images['magenta'][:, :, 2] = \
                    data[:, :, comps[i]]/data[:, :, comps].max()
            elif colors[i] == 'cyan':
                images['cyan'][:, :, 1] = \
                    data[:, :, comps[i]]/data[:, :, comps].max()
                images['cyan'][:, :, 2] = \
                    data[:, :, comps[i]]/data[:, :, comps].max()
            else:
                raise ValueError("Unknown color. Must be red, green, blue, "
                                 "yellow, magenta, or cyan.")
        else:
            raise ValueError("Unknown normalization method."
                             "Must be 'individual' or 'global'.")

    for i in colors:
        images['rgb'] += images[i]

    if display:
        fig, ax = plt.subplots(1)
        ax.imshow(images['rgb'])
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, images
    return images
