import matplotlib.pylab as plt
import numpy as np
from skimage import filters, measure
from scipy import ndimage
from scipy.stats.mstats import gmean
from astropy.stats import bootstrap


def remove_particles(segmentation):
    numpoints = segmentation.max()
    removed_any = False
    plt.figure(frameon=False)
    plt.imshow(segmentation, cmap='nipy_spectral')
    plt.title('Choose particles to remove...')
    coords = np.array(plt.ginput(numpoints, timeout=0, show_clicks=True))
    plt.close()
    n_removed = len(coords)
    if n_removed > 0:
        removed_any = True
    for i in range(0, len(coords)):
        val = segmentation[np.int32(coords[i][1]), np.int32(coords[i][0])]
        segmentation[segmentation == val] = 0
    return segmentation, removed_any, n_removed


def pick_particles(segmentation):
    numpoints = segmentation.max()
    # warnings.filterwarnings('ignore')
    plt.figure(frameon=False)
    plt.imshow(segmentation, cmap='nipy_spectral')
    plt.title('Choose particles ...')
    coords = np.array(plt.ginput(numpoints, timeout=0, show_clicks=True))
    plt.close()

    particles = np.zeros_like(segmentation)
    for i in range(0, len(coords)):
        val = segmentation[np.int32(coords[i][1]), np.int32(coords[i][0])]
        particles[segmentation == val] = val
    return particles


def preprocess(s, thresh=None, border=5):
    im_filt = ndimage.median_filter(s.data, 3)

    if not thresh:
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


def get_props(s, cutoff=None, thresh=None, border=5):
    results = {}
    results['filename'] = s.metadata.General.original_filename
    results['diameters'] = np.array([])
    results['max_ferets'] = np.array([])
    results['min_ferets'] = np.array([])
    results['removed_any'] = False
    results['n_removed'] = 0
    results['n_measured'] = 0

    if s.axes_manager[0].units == 'nm':
        pixsize = s.axes_manager[0].scale
    elif s.axes_manager[0].units == 'µm':
        pixsize = 1000 * s.axes_manager[0].scale
    else:
        raise ValueError('Unknown spatial units in image')
    segmentation = preprocess(s, thresh, border)

    particles, results['removed_any'], results['n_removed'] = \
        remove_particles(segmentation)

    props = measure.regionprops(particles, coordinates='xy')
    results['n_measured'] = len(props)

    for i in range(0, len(props)):
        d = pixsize * props[i]['equivalent_diameter']
        maxf = pixsize * props[i]['major_axis_length']
        minf = pixsize * props[i]['minor_axis_length']

        results['diameters'] = np.append(results['diameters'], d)
        results['max_ferets'] = np.append(results['max_ferets'], maxf)
        results['min_ferets'] = np.append(results['min_ferets'], minf)

    return results


def bootstrap_analysis(results, props):
    def test_statistic(x):
        return np.mean(x), np.median(x)

    data = {}
    output = {}

    if type(results) is list:
        if type(props) is list:
            for j in props:
                data[j] = np.concatenate([i[j] for i in results])
        else:
            data[props] = np.concatenate([i[props] for i in results])
    elif type(results) is dict:
        if type(props) is list:
            for j in props:
                data[j] = results[j]
        else:
            data[props] = results[props]

    for i in data.keys():
        output[i + "_bootstrap"] = {}
        result = bootstrap(data[i], 1000, bootfunc=test_statistic)
        output[i + "_bootstrap"]['Mean95'] = 2 * np.std(result[:, 0])
        output[i + "_bootstrap"]['Median95'] = 2 * np.std(result[:, 1])
    return output


def get_geometric_stats(data, verbose=False):
    data_gmean = gmean(data)
    data_gstd = np.exp(np.sqrt((np.log(data / data_gmean)**2).sum() / len(data)))
    if verbose:
        print('Geometric Mean: %.2f' % data_gmean)
        print('Geometric StdDev: %.2f' % data_gstd)
    return data_gmean, data_gstd


def csd(data, plot=False, outfile=None):
    max_val = np.int32(np.ceil(data.max()))
    csd = np.zeros([max_val, 2])
    csd[:, 0] = np.arange(0, max_val)
    for i in range(0, max_val):
        csd[i, 1] = sum(data < i)
    if plot:
        plt.figure()
        plt.scatter(csd[:, 0], csd[:, 1])
    if outfile:
        np.savetxt(outfile, csd, header='Diameter\tCSD\n(nm)\tN.A.',
                   fmt='%.1f', delimiter='\t')
    return csd
