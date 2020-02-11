import numpy as np
import matplotlib.pylab as plt


def get_mask(s, r1, r2=None, type='circular'):

    h, w = s.data.shape
    center = [np.int32(s.data.shape[0] / 2), np.int32(s.data.shape[1] / 2)]

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    if type == 'circular':
        mask = dist_from_center <= r1
    elif type == 'annular':
        mask = np.logical_and(dist_from_center <= r2, dist_from_center > r1)
    else:
        raise ValueError("Unknow mask type %s. Must be 'circle' or 'annular'."
                         % type)
    return mask


def display_masks(im, masks, alpha=0.2, log=False):
    colors = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']
    plt.figure()
    if log:
        plt.imshow(np.log(im.data + 1))
    else:
        plt.imshow(im.data)
    idx = 0
    for i in masks:
        plt.imshow(i, alpha=alpha, cmap=colors[idx])
        idx += 1
    return
