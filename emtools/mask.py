import numpy as np
import matplotlib.pylab as plt


def get_mask(s, r, r_outer=None, type='circular'):
    """
    Create a 2D circular or annular logical mask.

    Args
    ----------
    s : Hyperspy Signal2D
        2D data to mask
    r : int
        Radius of the mask. If 'type' is circular, this defines the circle
        radius.  If 'annular', this defines the inner radius.
    r_outer : int
        For 'type' annular, defines the outer radius of the mask.
    type : str
        Must be either 'circular' or 'annular'


    Returns
    ----------
    mask : Numpy array
        2D logical mask

    """
    h, w = s.data.shape
    center = [np.int32(s.data.shape[0] / 2), np.int32(s.data.shape[1] / 2)]

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    if type == 'circular':
        mask = dist_from_center <= r
    elif type == 'annular':
        mask = np.logical_and(dist_from_center <= r_outer,
                              dist_from_center > r)
    else:
        raise ValueError("Unknow mask type %s. Must be 'circle' or 'annular'."
                         % type)
    return mask


def display_masks(im, masks, alpha=0.2, log=False):
    """
    Overlay mask on image for visualization.

    Args
    ----------
    im : Hyperspy Signal2D
        Image or pattern to display
    masks : list
        Iterable of masks to overlay on image or pattern
    alpha : float
        Degree of transparency of the masks.  Must be between 0.0 and 1.0
    log : bool
        If True, display image on a log scale (useful for diffraction patterns)

    Returns
    ----------

    """
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
