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
    if len(data.data.shape) == 3:
        scaleX = data.axes_manager[1].scale
        scaleY = data.axes_manager[0].scale
        scaleZ = data.axes_manager[2].scale

        offsetX = data.axes_manager[1].offset
        offsetY = data.axes_manager[0].offset
        offsetZ = data.axes_manager[2].offset

        label_image = label(data.data > threshold * data.data.max())
        label_image = hs.signals.Signal2D(label_image)
        regions = regionprops(label_image.data)

        data.original_metadata.points = \
            np.asarray([(i.centroid) for i in regions])
        data.original_metadata.points_cal = data.original_metadata.points * \
            [scaleY, scaleX, scaleZ] + [offsetY, offsetZ, offsetX]

    elif len(data.data.shape) == 2:
        scaleX = data.axes_manager[1].scale
        scaleY = data.axes_manager[0].scale

        offsetX = data.axes_manager[1].offset
        offsetY = data.axes_manager[0].offset

        label_image = label(data.data > threshold * data.data.max())
        label_image = hs.signals.Signal2D(label_image)
        regions = regionprops(label_image.data)

        data.original_metadata.points = \
            np.asarray([(i.centroid) for i in regions])
        data.original_metadata.points_cal = data.original_metadata.points * \
            [scaleX, scaleY] + [offsetX, offsetY]

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


def get_surface(data, blur_sigma=3, canny_sigma=0.1):
    out = data.deepcopy()
    blur = gaussian(data.data, sigma=blur_sigma)
    thresh_val = threshold_otsu(blur)
    thresholded = blur > thresh_val
    edges = np.zeros_like(thresholded.data)
    if len(data.data.shape) == 3:
        for i in range(thresholded.shape[0]):
            edges[i, :, :] = canny(thresholded[i, :, :], sigma=canny_sigma)
    else:
        edges = canny(thresholded, sigma=canny_sigma)

    out.data = edges
    return out


def output_stats(mindistance):
    print('Statistical Output')
    print('------------------------')
    print('Number of particles measured: %s' % str(len(mindistance)))
    print('Mean distance (nm): %.1f' %
          float(mindistance.mean()))
    print('Standard Deviation (nm): %.1f' %
          float(mindistance.std()))
    print('Minimum measured distance (nm): %.1f' %
          float(mindistance.min()))
    print('Maximum measured distance (nm): %.1f' %
          float(mindistance.max()))
    return


def distance_calc(surface, data, print_stats=False):
    points = data.original_metadata.points
    scaleX = data.axes_manager[1].scale
    scaleY = data.axes_manager[0].scale
    offsetX = data.axes_manager[1].offset
    offsetY = data.axes_manager[0].offset
    mindistance = np.zeros(len(points))
    surfacepoints = np.array(np.where(surface.data)).T
    if len(data.data.shape) == 3:
        scaleZ = data.axes_manager[2].scale
        offsetZ = data.axes_manager[2].offset
        minloc = np.zeros([len(points), 3])
        for i in range(0, len(points)):
            distance = np.sqrt(((surfacepoints - points[i])**2).sum(1))
            mindistance[i] = distance.min()
            minindex = np.argmin(distance)
            minloc[i, :] = surfacepoints[minindex, :]

        minloc_cal = minloc * [scaleY, scaleX, scaleZ] + \
            [offsetY, offsetZ, offsetX]
        mindistance_cal = mindistance * scaleX

    elif len(data.data.shape) == 2:
        minloc = np.zeros([len(points), 2])
        for i in range(0, len(points)):
            distance = np.sqrt(((surfacepoints - points[i])**2).sum(1))
            mindistance[i] = distance.min()
            minindex = np.argmin(distance)
            minloc[i, :] = surfacepoints[minindex, :]
        minloc_cal = minloc * [scaleX, scaleY] + [offsetX, offsetY]
        mindistance_cal = mindistance * scaleX
    if print_stats:
        output_stats(mindistance, scaleX)

    data.original_metadata.minloc = minloc
    data.original_metadata.minloc_cal = minloc_cal
    data.original_metadata.mindistance = mindistance
    data.original_metadata.mindistance_cal = mindistance_cal
    return data


def plot_result(data_signal, edge_signal, idx=None,
                display='edges', axis='XZ'):
    if len(data_signal.data.shape) == 3:
        particle_loc = data_signal.original_metadata.points_cal[idx]
        edge_loc = data_signal.original_metadata.minloc_cal[idx]

        if axis == 'XZ' or axis == 'ZX':
            imageXZ = data_signal.inav[edge_loc[0]]
            imageXZ.data[edge_signal.inav[edge_loc[0]].data] = \
                1.1 * imageXZ.data.max()
            imageXZ.plot(cmap='gray')
            ax = plt.gca()
            ax.plot([particle_loc[2], edge_loc[2]],
                    [particle_loc[1], edge_loc[1]],
                    '-wo')
            ax.plot(edge_loc[2], edge_loc[1],
                    'o', color='b', alpha=0.5)
            ax.plot(particle_loc[2], particle_loc[1],
                    'o', color='r', alpha=0.5)

        elif axis == 'XY' or axis == 'YX':
            imageXY = data_signal.isig[:, edge_loc[1]].as_signal2D((1, 0))
            imageXY.data[edge_signal.isig[:, edge_loc[1]].data] = \
                1.1 * imageXY.data.max()
            imageXY.plot(cmap='gray')
            ax = plt.gca()

            ax.plot([particle_loc[2], edge_loc[2]],
                    [particle_loc[0], edge_loc[0]],
                    '-wo')
            ax.plot(edge_loc[2], edge_loc[0],
                    'o', color='b')
            ax.plot(particle_loc[2], particle_loc[0],
                    'o', color='r')

        elif axis == 'YZ' or axis == 'ZY':
            imageYZ = data_signal.isig[edge_loc[2], :].as_signal2D((1, 0))
            imageYZ.data[edge_signal.isig[edge_loc[2], :].data] = \
                1.1 * imageYZ.data.max()
            imageYZ.plot(cmap='gray')
            ax = plt.gca()
            ax.plot([particle_loc[1], edge_loc[1]],
                    [particle_loc[0], edge_loc[0]],
                    '-wo')
            ax.plot(edge_loc[1], edge_loc[0], 'o', color='b', alpha=0.5)
            ax.plot(particle_loc[1], particle_loc[0],
                    'o', color='r', alpha=0.5)

        elif axis == 'XYall' or 'XZall' or 'YZall':
            particle_loc = data_signal.original_metadata.points_cal
            edge_loc = data_signal.original_metadata.minloc_cal
            if axis == 'XZall':
                maximageXZ = data_signal.max()
                maximageXZ.plot(cmap='afmhot')
                ax = plt.gca()

                _ = ax.plot([particle_loc[:, 2], edge_loc[:, 2]],
                            [particle_loc[:, 1], edge_loc[:, 1]],
                            '-o')

            elif axis == 'XYall':
                maximageXY = data_signal.max(2).as_signal2D((1, 0))
                maximageXY.plot(cmap='afmhot')
                ax = plt.gca()

                _ = ax.plot([particle_loc[:, 2], edge_loc[:, 2]],
                            [particle_loc[:, 0], edge_loc[:, 0]],
                            '-o')

            elif axis == 'YZall':
                maximageXY = data_signal.max(1).as_signal2D((1, 0))
                maximageXY.plot(cmap='afmhot')
                ax = plt.gca()

                _ = ax.plot([particle_loc[:, 1], edge_loc[:, 1]],
                            [particle_loc[:, 0], edge_loc[:, 0]],
                            '-o')
        else:
            raise ValueError('Unknown axis: %s' % axis)

    elif len(data_signal.data.shape) == 2:
        plot_signal = data_signal.deepcopy()
        particle_loc = data_signal.original_metadata.points_cal
        edge_loc = data_signal.original_metadata.minloc_cal

        plot_signal.data[edge_signal.data] = 1.1 * plot_signal.data.max()
        plot_signal.plot(cmap='gray')
        ax = plt.gca()

        _ = ax.plot([particle_loc[:, 1], edge_loc[:, 1]],
                    [particle_loc[:, 0], edge_loc[:, 0]],
                    '-o')
    fig = plt.gcf()
    return fig


def get_particle_distances(stack, verbose=True):
    if verbose:
        print('Locating particle centroids...')
        stack = threshold_particles(stack, threshold=0.5, return_labels=False)
        print('Done!')
        print('Locating surface of volume...')
        edges = get_surface(stack, blur_sigma=3, canny_sigma=0.1)
        print('Done!')
        print('Calculating particle-surface distances...')
        stack = distance_calc(edges, stack, print_stats=False)
        print('Done!')
        output_stats(stack.original_metadata.mindistance_cal)
    else:
        stack = threshold_particles(stack, threshold=0.5, return_labels=False)
        edges = get_surface(stack, blur_sigma=3, canny_sigma=0.1)
        stack = distance_calc(edges, stack, print_stats=False)

    return stack, edges
