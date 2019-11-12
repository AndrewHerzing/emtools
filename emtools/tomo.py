# -*- coding: utf-8 -*-
#
# This file is part of EMTools

"""
Tomo module for EMTools package

@author: Andrew Herzing
"""

import numpy as np
import matplotlib.pylab as plt
import hyperspy.api as hs
from skimage.feature import peak_local_max, canny
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from scipy import signal


def template_match(data, template, threshold=0.5):
    """
    Locate particles in volume using template matching.

    Args
    ----------
    data : Hyperspy Signal2D
        Volumetric data containing particles
    template : Hyperspy Signal2D
        Particle template to match to data
    threshold : float
        Fraction of the maximum value of the template matching result. Regions
        of the matching result above this value are identified as particles.

    Returns
    ----------
    data : Hyperspy Signal2D
        Modified version of the input data with the centroid location added to
        the metadata.  Absolute coordinates and calibrated positions are stored
        in data.original_metadata.points and data.original_metadata.points_cal,
        respectively.

    """
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
    """
    Locate particles in volume using binary sementation.

    Args
    ----------
    data : Hyperspy Signal2D
        Volumetric data containing particles
    threshold : float
        Fraction of the maximum value of the data to use as segmentation
        threshold.
    return_labels : bool
        If True, return the data with particle locations, the segmentation
        image, and the region properties.  If False, retun only the data with
        particle locations.  Default is False.

    Returns
    ----------
    data : Hyperspy Signal2D
        Modified version of the input data with the centroid location added to
        the metadata.  Absolute coordinates and calibrated positions are stored
        in data.original_metadata.points and data.original_metadata.points_cal,
        respectively.
    label_image : Hyperspy Signal2D
        Segmented image with all idenitified particle regions assigned a unique
        grayscale value.
    regions : list
        Results of region_props analysis of each individual labeled particle.

    """
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
    """
    Plot data with single particle location indicated.

    Args
    ----------
    data : Hyperspy Signal2D
        Volumetric data containing particles and their locations.
    index : int
        Particle location to plot.

    """
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
    """
    Locate the surface of the agglomerate in a reconstructed tomogram.

    Args
    ----------
    data : Hyperspy Signal2D
        Volumetric data
    blur_sigma : float
        Sigma value to provide the Gaussian blur function.
    canny_sigma : float
        Sigma value to provide the Canny edge detection function.

    Returns
    ----------
    edges : Hyperspy Signal2D
        Copy of the input data after edge detection.  Underlying data is a
        boolean array, with True values at the edge of the agglomerate and
        False values everywhere else.

    """
    edges = data.deepcopy()
    blur = gaussian(data.data, sigma=blur_sigma)
    thresh_val = threshold_otsu(blur)
    thresholded = blur > thresh_val
    edges.data = np.zeros_like(thresholded.data)
    if len(data.data.shape) == 3:
        for i in range(thresholded.shape[0]):
            edges.data[i, :, :] = canny(thresholded[i, :, :],
                                        sigma=canny_sigma)
    else:
        edges.data = canny(thresholded, sigma=canny_sigma)

    return edges


def output_stats(mindistance):
    """
    Print statistics of an array to the terminal.

    Args
    ----------
    mindistance : Numpy array
        Calculated minimum distance to the surface for all particles.

    """
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


def surface_distance_calc(surface, data, print_stats=False):
    """
    Calculate distance of all particles from the agglomerate surface.

    Args
    ----------
    surface : Hyperspy Signal2D
        Signal containing the indicating the location of the agglomerate
        surface.
    data : Hyperspy Signal2D
        Volumetric data containing particle locations
    print_stats : bool
        If True, print statistical properties of the calculated distances.

    Returns
    ----------
    data : Hyperspy Signal2D
        Modified version of the input data with the calculated distances and
        the location of the nearest surface location added to the metadata.
        Absolute distances and calibrated distances are stored in
        data.original_metadata.mindistance and
        data.original_metadata.mindistance_cal, respectively. Absolute
        coordinates of the surface locations and the calibrated coordinates are
        stored in data.original_metadata.minloc and
        data.original_metadata.minloc_cal, respectively.

    """
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


def plot_particle_to_particle_result(data_signal, idx=None, axis='XZ'):
    """
    Plot particle distance measurement results overlain with tomogram.

    Args
    ----------
    data_signal : Hyperspy Signal2D
        Tomographic reconstruction with particle distance results in metadata.
    idx : int or None
        Index of the particle to be displayed.
    axis : str
        Axis along which to integrate the tomogram for display.  If 'axis' is
        one of the following: 'XZ', 'YZ', 'XY', 'YX', 'ZY', 'ZX', the
        nearest slice along this axis in the tomogram to the point defined by
        'idx' is displayed along with the identified surface pixels.  If 'axis'
        is 'XZall', 'YZall', or 'XYall', the tomogram is integrated along the
        axis and all points, minimum surface distances, and the projected edge
        pixels are displayed.

    Returns
    ----------
    fig : Matplotlib Figure

    """
    if len(data_signal.data.shape) == 3:
        particle_loc = data_signal.original_metadata.points_cal[idx]
        closest_particle_loc = \
            data_signal.original_metadata.particle_locs_cal[idx]

        if axis == 'XZ' or axis == 'ZX':
            imageXZ = data_signal.inav[closest_particle_loc[0]]
            imageXZ.plot(cmap='gray')
            ax = plt.gca()
            ax.plot([particle_loc[2], closest_particle_loc[2]],
                    [particle_loc[1], closest_particle_loc[1]],
                    '-wo')
            ax.plot(closest_particle_loc[2], closest_particle_loc[1],
                    'o', color='b', alpha=0.5)
            ax.plot(particle_loc[2], particle_loc[1],
                    'o', color='r', alpha=0.5)
            plt.title('XZ Slice')

        elif axis == 'XY' or axis == 'YX':
            imageXY = data_signal.isig[:, closest_particle_loc[1]]
            imageXY = imageXY.as_signal2D((1, 0))
            imageXY.plot(cmap='gray')
            ax = plt.gca()
            ax.plot([particle_loc[2], closest_particle_loc[2]],
                    [particle_loc[0], closest_particle_loc[0]],
                    '-wo')
            ax.plot(closest_particle_loc[2], closest_particle_loc[0],
                    'o', color='b', alpha=0.5)
            ax.plot(particle_loc[2], particle_loc[0],
                    'o', color='r', alpha=0.5)
            plt.title('XY Slice')

        elif axis == 'YZ' or axis == 'ZY':
            imageYZ = data_signal.isig[closest_particle_loc[2], :]
            imageYZ = imageYZ.as_signal2D((1, 0))
            imageYZ.plot(cmap='gray')
            ax = plt.gca()
            ax.plot([particle_loc[1], closest_particle_loc[1]],
                    [particle_loc[0], closest_particle_loc[0]],
                    '-wo')
            ax.plot(closest_particle_loc[1], closest_particle_loc[0],
                    'o', color='b', alpha=0.5)
            ax.plot(particle_loc[1], particle_loc[0],
                    'o', color='r', alpha=0.5)
            plt.title('YZ Slice')

        elif axis == 'XYall' or 'XZall' or 'YZall':
            particle_loc = data_signal.original_metadata.points_cal
            closest_particle_locs = \
                data_signal.original_metadata.particle_locs_cal
            if axis == 'XZall':
                maximageXZ = data_signal.max()
                maximageXZ.plot(cmap='afmhot')
                ax = plt.gca()

                ax.plot([particle_loc[:, 2], closest_particle_locs[:, 2]],
                        [particle_loc[:, 1], closest_particle_locs[:, 1]],
                        '-o')
                plt.title('XZ Projection')

            elif axis == 'XYall':
                maximageXY = data_signal.max(2).as_signal2D((1, 0))
                maximageXY.plot(cmap='afmhot')
                ax = plt.gca()

                ax.plot([particle_loc[:, 2], closest_particle_locs[:, 2]],
                        [particle_loc[:, 0], closest_particle_locs[:, 0]],
                        '-o')
                plt.title('XY Projection')

            elif axis == 'YZall':
                maximageXY = data_signal.max(1).as_signal2D((1, 0))
                maximageXY.plot(cmap='afmhot')
                ax = plt.gca()

                ax.plot([particle_loc[:, 1], closest_particle_locs[:, 1]],
                        [particle_loc[:, 0], closest_particle_locs[:, 0]],
                        '-o')
                plt.title('YZ Projection')
        else:
            raise ValueError('Unknown axis: %s' % axis)

    elif len(data_signal.data.shape) == 2:
        plot_signal = data_signal.deepcopy()
        particle_loc = data_signal.original_metadata.points_cal
        closest_particle_loc = data_signal.original_metadata.particle_locs_cal

        plot_signal.plot(cmap='gray')
        ax = plt.gca()

        ax.plot([particle_loc[:, 1], closest_particle_loc[:, 1]],
                [particle_loc[:, 0], closest_particle_loc[:, 0]],
                '-o')
    fig = plt.gcf()
    return fig


def plot_particle_to_surface_result(data_signal, edge_signal, idx=None,
                                    axis='XZ'):
    """
    Plot particle distance measurement results overlain with tomogram.

    Args
    ----------
    data_signal : Hyperspy Signal2D
        Tomographic reconstruction with particle distance results in metadata.
    edge_signal : Hyperspy Signal2D
        Signal containing location of the particle surface after Canny edge
        processing.
    idx : int or None
        Index of the particle to be displayed.
    axis : str
        Axis along which to integrate the tomogram for display.  If 'axis' is
        one of the following: 'XZ', 'YZ', 'XY', 'YX', 'ZY', 'ZX', the
        nearest slice along this axis in the tomogram to the point defined by
        'idx' is displayed along with the identified surface pixels.  If 'axis'
        is 'XZall', 'YZall', or 'XYall', the tomogram is integrated along the
        axis and all points, minimum surface distances, and the projected edge
        pixels are displayed.

    Returns
    ----------
    fig : Matplotlib Figure

    """
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
            plt.title('XZ Slice')

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
            plt.title('XY Slice')

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
            plt.title('YZ Slice')

        elif axis == 'XYall' or 'XZall' or 'YZall':
            particle_loc = data_signal.original_metadata.points_cal
            edge_loc = data_signal.original_metadata.minloc_cal
            if axis == 'XZall':
                maximageXZ = data_signal.max()
                maximageXZ.plot(cmap='afmhot')
                ax = plt.gca()

                ax.plot([particle_loc[:, 2], edge_loc[:, 2]],
                        [particle_loc[:, 1], edge_loc[:, 1]],
                        '-o')
                plt.title('XZ Projection')

            elif axis == 'XYall':
                maximageXY = data_signal.max(2).as_signal2D((1, 0))
                maximageXY.plot(cmap='afmhot')
                ax = plt.gca()

                ax.plot([particle_loc[:, 2], edge_loc[:, 2]],
                        [particle_loc[:, 0], edge_loc[:, 0]],
                        '-o')
                plt.title('XY Projection')

            elif axis == 'YZall':
                maximageXY = data_signal.max(1).as_signal2D((1, 0))
                maximageXY.plot(cmap='afmhot')
                ax = plt.gca()

                ax.plot([particle_loc[:, 1], edge_loc[:, 1]],
                        [particle_loc[:, 0], edge_loc[:, 0]],
                        '-o')
                plt.title('YZ Projection')
        else:
            raise ValueError('Unknown axis: %s' % axis)

    elif len(data_signal.data.shape) == 2:
        plot_signal = data_signal.deepcopy()
        particle_loc = data_signal.original_metadata.points_cal
        edge_loc = data_signal.original_metadata.minloc_cal

        plot_signal.data[edge_signal.data] = 1.1 * plot_signal.data.max()
        plot_signal.plot(cmap='gray')
        ax = plt.gca()

        ax.plot([particle_loc[:, 1], edge_loc[:, 1]],
                [particle_loc[:, 0], edge_loc[:, 0]],
                '-o')
    fig = plt.gcf()
    return fig


def get_particle_to_surface_distances(stack, verbose=True, threshold=0.5):
    """
    Perform the entire particle-surface distance workflow.

    Args
    ----------
    stack : Hyperspy Signal2D or TomoStack
        Tomographic reconstruction to be analyzed.
    verbose : str
        If True, progress updates at each point and the measurement results
        to the terminal. Default is True.
    threshold : float
        Fraction of the maximum value of the template matching result. Regions
        of the matching result above this value are identified as particles.

    Returns
    ----------
    stack : Hyperspy Signal2D
        Modified version of the input data with the calculated distances and
        the location of the nearest surface location added to the metadata.
        Absolute distances and calibrated distances are stored in
        data.original_metadata.mindistance and
        data.original_metadata.mindistance_cal, respectively. Absolute
        coordinates of the surface locations and the calibrated coordinates are
        stored in data.original_metadata.minloc and
        data.original_metadata.minloc_cal, respectively.
    edges : Hyperspy Signal2D
        Signal with the same dimensions as 'stack' where data is boolean array
        indicating the surface pixel locations.


    """
    for i in range(len(stack.data.shape)):
        stack.axes_manager[i].offset = 0
    if verbose:
        print('Locating particle centroids...')
        stack = threshold_particles(stack, threshold=threshold,
                                    return_labels=False)
        print('Done!')
        print('Locating surface of volume...')
        edges = get_surface(stack, blur_sigma=3, canny_sigma=0.1)
        print('Done!')
        print('Calculating particle-surface distances...')
        stack = surface_distance_calc(edges, stack, print_stats=False)
        print('Done!')
        output_stats(stack.original_metadata.mindistance_cal)
    else:
        stack = threshold_particles(stack, threshold=threshold,
                                    return_labels=False)
        edges = get_surface(stack, blur_sigma=3, canny_sigma=0.1)
        stack = surface_distance_calc(edges, stack, print_stats=False)

    return stack, edges


def get_particle_to_particle_distances(stack, verbose=True, threshold=0.5):
    """
    Perform the entire particle-particle distance workflow.

    Args
    ----------
    stack : Hyperspy Signal2D or TomoStack
        Tomographic reconstruction to be analyzed.
    verbose : str
        If True, progress updates at each point and the measurement results
        to the terminal. Default is True.
    threshold : float
        Fraction of the maximum value of the template matching result. Regions
        of the matching result above this value are identified as particles.

    Returns
    ----------
    stack : Hyperspy Signal2D
        Modified version of the input data with the calculated distances and
        the location of the nearest surface location added to the metadata.
        Absolute distances and calibrated distances are stored in
        data.original_metadata.mindistance and
        data.original_metadata.mindistance_cal, respectively. Absolute
        coordinates of the surface locations and the calibrated coordinates are
        stored in data.original_metadata.minloc and
        data.original_metadata.minloc_cal, respectively.
    edges : Hyperspy Signal2D
        Signal with the same dimensions as 'stack' where data is boolean array
        indicating the surface pixel locations.


    """
    for i in range(len(stack.data.shape)):
        stack.axes_manager[i].offset = 0
    if verbose:
        print('Locating particle centroids...')
        stack = threshold_particles(stack, threshold=threshold,
                                    return_labels=False)
        print('Done!')
        print('Calculating particle-particle distances...')
        stack = particle_distance_calc(stack, print_stats=False)
        print('Done!')
        output_stats(stack.original_metadata.particle_distances_cal)
    else:
        stack = threshold_particles(stack, threshold=threshold,
                                    return_labels=False)
        stack = particle_distance_calc(stack, print_stats=False)

    return stack


def particle_distance_calc(data, print_stats=False):
    """
    Calculate distance of all particles from the agglomerate surface.

    Args
    ----------
    data : Hyperspy Signal2D
        Volumetric data containing particle locations
    print_stats : bool
        If True, print statistical properties of the calculated distances.

    Returns
    ----------
    data : Hyperspy Signal2D
        Modified version of the input data with the calculated
        particle-particledistances added to the metadata.
        Absolute distances and calibrated distances are stored in
        data.original_metadata.particle_distances and
        data.original_metadata.particle_distances_cal, respectively.

    """
    points = data.original_metadata.points
    scaleX = data.axes_manager[1].scale
    scaleY = data.axes_manager[0].scale
    offsetX = data.axes_manager[1].offset
    offsetY = data.axes_manager[0].offset
    mindistance = np.zeros(len(points))

    if len(data.data.shape) == 3:
        scaleZ = data.axes_manager[2].scale
        offsetZ = data.axes_manager[2].offset
        minloc = np.zeros([len(points), 3])
        for i in range(0, len(points)):
            distance = np.sqrt(((points - points[i])**2).sum(1))
            distance_sorted = np.sort(distance)
            mindistance[i] = distance_sorted[1:].min()
            minindex = np.where(distance == mindistance[i])
            minloc[i, :] = points[minindex, :]

        minloc_cal = minloc * [scaleY, scaleX, scaleZ] + \
            [offsetY, offsetZ, offsetX]
        mindistance_cal = mindistance * scaleX

    elif len(data.data.shape) == 2:
        minloc = np.zeros([len(points), 2])
        for i in range(0, len(points)):
            distance = np.sqrt(((points - points[i])**2).sum(1))
            distance_sorted = np.sort(distance)
            mindistance[i] = distance_sorted[1:].min()
            minindex = np.where(distance == mindistance[i])
            minloc[i, :] = points[minindex, :]
        minloc_cal = minloc * [scaleX, scaleY] + [offsetX, offsetY]
        mindistance_cal = mindistance * scaleX
    if print_stats:
        output_stats(mindistance, scaleX)

    data.original_metadata.particle_distances = mindistance
    data.original_metadata.particle_distances_cal = mindistance_cal
    data.original_metadata.particle_locs = minloc
    data.original_metadata.particle_locs_cal = minloc_cal
    return data
