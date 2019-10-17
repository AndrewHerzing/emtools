import numpy as np
import matplotlib.pylab as plt
import hyperspy.api as hs
from skimage.feature import peak_local_max, canny
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from scipy import signal


def template_match(data, template, threshold=0.5):
    scaleX = data.axes_manager[1].scale
    scaleY = data.axes_manager[0].scale
    scaleZ = data.axes_manager[2].scale

    offsetX = data.axes_manager[1].offset
    offsetY = data.axes_manager[0].offset
    offsetZ = data.axes_manager[2].offset

    if len(data.data.shape) != len(template.data.shape):
        raise ValueError('Data shape %s and template shape %s are inconsistent'
                         % (data.data.shape, template.data.shape))

    img = data.data - data.data.mean()
    template.data = template.data - template.data.mean()

    result = signal.fftconvolve(img, template, mode='same')

    threshold = result.max() * threshold
    points = np.float32(peak_local_max(
        result, threshold_abs=threshold)) - [1, 1, -1]

    data.original_metadata.points = points
    data.original_metadata.points_cal = points * [scaleY, scaleX, scaleZ] +\
        [offsetY, offsetZ, offsetX]

    return data


def threshold_particles(data, threshold=0.5, return_labels=False):
    scaleX = data.axes_manager[1].scale
    scaleY = data.axes_manager[0].scale
    scaleZ = data.axes_manager[2].scale

    offsetX = data.axes_manager[1].offset
    offsetY = data.axes_manager[0].offset
    offsetZ = data.axes_manager[2].offset

    label_image = label(data.data > threshold * data.data.max())
    label_image = hs.signals.Signal2D(label_image)
    regions = regionprops(label_image.data)

    data.original_metadata.points = np.asarray([(i.centroid) for i in regions])
    data.original_metadata.points_cal = data.original_metadata.points * \
        [scaleY, scaleX, scaleZ] + [offsetY, offsetZ, offsetX]

    if return_labels:
        return data, label_image, regions
    else:
        return data


def plot_points(data, index):
    if 'points' in data.original_metadata.keys():
        points = data.original_metadata.points
    else:
        raise ValueError('No points found in metadata')

    if np.where(np.abs([i[0] - index for i in points]) < 1)[0].any():
        loc = np.where(np.abs([i[0] - index for i in points]) < 1)[0]
        plt.figure()
        plt.imshow(data.inav[index].data)
        for i in range(0, len(loc)):
            plt.scatter(points[loc[i]][2], points[loc[i]][1], s=50,
                        c='red', marker='o', alpha=0.3)
    else:
        nearest = points[np.abs([i[0] - index for i in points]).argmin()][0]
        print('No maxima found at this slice')
        print('Nearest slice with maxima is: %s' % str(nearest))
    return


def plot_all_points(data, axes=['XZ', 'XY'], cmap='inferno'):
    if 'points_cal' in data.original_metadata.keys():
        points_cal = data.original_metadata.points_cal
    else:
        raise ValueError('No points found in metadata')
    maximageXZ = data.max()

    maximageXY = data.max(2).as_signal2D((1, 0))

    axes = hs.plot.plot_images([maximageXZ, maximageXY],
                               per_row=2, scalebar='all',
                               colorbar=None, axes_decor='off', cmap=cmap)
    axes[0].plot(points_cal[:, 2], points_cal[:, 1], 'ro', markersize=2)
    axes[1].plot(points_cal[:, 2], points_cal[:, 0], 'ro', markersize=2)

    return axes


def get_surface(data, blur_sigma=3, canny_sigma=0.1):
    blur = gaussian(data.data, sigma=blur_sigma)
    thresh_val = threshold_otsu(blur)
    thresholded = blur > thresh_val
    edges = np.zeros_like(thresholded.data)
    for i in range(thresholded.shape[0]):
        edges[i, :, :] = canny(thresholded[i, :, :], sigma=canny_sigma)
    edges = hs.signals.Signal2D(edges)
    return edges


def distance_calc(surface, data, print_stats=False):
    points = data.original_metadata.points
    scale = data.axes_manager[1].scale
    mindistance = np.zeros(len(points))
    minloc = np.zeros([len(points), 3])
    surfacepoints = np.array(np.where(surface.data)).T
    for i in range(0, len(points)):
        distance = np.sqrt(((surfacepoints - points[i])**2).sum(1))
        mindistance[i] = distance.min()
        minindex = np.argmin(distance)
        minloc[i, :] = surfacepoints[minindex, :]

    if print_stats:
        print('Statistical Output')
        print('------------------------')
        print('Mean distance (nm): %.1f' %
              float(scale * mindistance.mean()))
        print('Standard Deviation (nm): %.1f' %
              float(scale * mindistance.std()))
        print('Minimum measured distance (nm): %.1f' %
              float(scale * mindistance.min()))
        print('Maximum measured distance (nm): %.1f' %
              float(scale * mindistance.max()))

    data.original_metadata.minloc = minloc
    data.original_metadata.mindistance = mindistance
    return data


def plot_result(edge_signal, idx, data_signal, display='edges', axis='XZ'):
    if display == 'edges':
        particle_loc = data_signal.original_metadata.points[idx]
        edge_loc = data_signal.original_metadata.minloc[idx]
        imageXZ = edge_signal.inav[edge_loc[0]]
        imageYZ = edge_signal.isig[edge_loc[2], :].as_signal2D((0, 1))
        imageXY = edge_signal.isig[:, edge_loc[1]].as_signal2D((1, 0))

    elif display == 'data':
        scaleY = data_signal.axes_manager[0].scale
        offsetY = data_signal.axes_manager[0].offset
        scaleX = data_signal.axes_manager[1].scale
        offsetX = data_signal.axes_manager[1].offset
        scaleZ = data_signal.axes_manager[2].scale
        offsetZ = data_signal.axes_manager[2].offset

        particle_loc = data_signal.original_metadata.points_cal[idx]
        edge_loc = data_signal.original_metadata.minloc[idx]
        edge_loc = [scaleY, scaleZ, scaleX] * edge_loc +\
            [offsetY, offsetZ, offsetX]

        imageXZ = data_signal.inav[edge_loc[0]]
        imageYZ = data_signal.isig[edge_loc[2], :].as_signal2D((1, 0))
        imageXY = data_signal.isig[:, edge_loc[1]].as_signal2D((1, 0))
    else:
        raise ValueError(
            "Unknown display option '%s'. Must be 'data' or 'edges'" % display)

    if axis == 'XZ' or axis == 'ZX':
        imageXZ.plot(cmap='gray')
        ax = plt.gca()
        ax.plot(edge_loc[2], edge_loc[1], 'o', color='b', alpha=0.5)
        ax.plot(particle_loc[2], particle_loc[1], 'o', color='r', alpha=0.5)
    elif axis == 'XY' or axis == 'YX':
        imageXY.plot(cmap='gray')
        ax = plt.gca()
        ax.plot(edge_loc[2], edge_loc[0], 'o', color='b', alpha=0.5)
        ax.plot(particle_loc[2], particle_loc[0], 'o', color='r', alpha=0.5)
    elif axis == 'YZ' or axis == 'ZY':
        imageYZ.plot(cmap='gray')
        ax = plt.gca()
        ax.plot(edge_loc[1], edge_loc[0], 'o', color='b', alpha=0.5)
        ax.plot(particle_loc[1], particle_loc[0], 'o', color='r', alpha=0.5)
    else:
        raise ValueError('Unknown axis: %s' % axis)
    return
