import matplotlib.pylab as plt
import numpy as np
from skimage import filters, measure
from scipy import ndimage


def remove_particles(segmentation):
    numpoints = segmentation.max()
    plt.figure(frameon=False)
    plt.imshow(segmentation, cmap='nipy_spectral')
    plt.title('Choose particles to remove...')
    coords = np.array(plt.ginput(numpoints, timeout=0, show_clicks=True))
    plt.close()

    for i in range(0, len(coords)):
        val = segmentation[np.int32(coords[i][1]), np.int32(coords[i][0])]
        segmentation[segmentation == val] = 0
    return segmentation


def pick_particles(segmentation):
    numpoints = segmentation.max()
    # warnings.filterwarnings('ignore')
    plt.figure(frameon=False)
    plt.imshow(segmentation, cmap='nipy_spectral')
    plt.title('Choose particles to remove...')
    coords = np.array(plt.ginput(numpoints, timeout=0, show_clicks=True))
    plt.close()

    particles = np.zeros_like(segmentation)
    for i in range(0, len(coords)):
        val = segmentation[np.int32(coords[i][1]), np.int32(coords[i][0])]
        particles[segmentation == val] = val
    return particles


def preprocess(s, border=5):
    im_filt = ndimage.median_filter(s.data, 3)

    thresh = filters.threshold_otsu(im_filt)
    im_thresh = im_filt < thresh

    im_thresh = ndimage.binary_erosion(im_thresh, iterations=5)
    im_thresh = ndimage.binary_dilation(im_thresh, iterations=5)

    im_labels = measure.label(im_thresh, background=0)

    for i in range(1, im_labels.max()):
        if np.any((im_labels == i)[0:border, :])\
            or np.any((im_labels == i)[:, 0:border])\
                or np.any((im_labels == i)[-border:, :])\
                or np.any((im_labels == i)[:, -border:]):
            im_labels[im_labels == i] = 0
    return im_labels


def get_props(s, pick=True):
    if s.axes_manager[0].units == 'nm':
        pixsize = s.axes_manager[0].scale
    elif s.axes_manager[0].units == 'µm':
        pixsize = 1000*s.axes_manager[0].scale
    else:
        raise ValueError('Unknown spatial units in image')
    segmentation = preprocess(s)

    if pick:
        particles = pick_particles(segmentation)
        props = measure.regionprops(particles, coordinates='xy')
    else:
        props = measure.regionprops(segmentation, coordinates='xy')
        new_props = []
        for i in range(0, len(props)):
            diameter = props[i]['equivalent_diameter']
            major = props[i]['major_axis_length']
            minor = props[i]['minor_axis_length']
            diff1 = np.abs((diameter-minor)/diameter)
            diff2 = np.abs((diameter-major)/diameter)
            if diff1 < 0.05 and diff2 < 0.05:
                new_props.append(props[i])
        props = new_props

    diameters = np.array([])
    max_ferets = np.array([])
    min_ferets = np.array([])

    for i in range(0, len(props)):
        d = pixsize*props[i]['equivalent_diameter']
        maxf = pixsize*props[i]['major_axis_length']
        minf = pixsize*props[i]['minor_axis_length']

        diameters = np.append(diameters, d)
        max_ferets = np.append(max_ferets, maxf)
        min_ferets = np.append(min_ferets, minf)

    return diameters, max_ferets, min_ferets
