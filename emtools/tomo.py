import numpy as np
import cv2
import matplotlib.pylab as plt
import hyperspy.api as hs
from skimage.feature import peak_local_max, canny
from skimage.filters import gaussian, threshold_otsu, sobel
from scipy import ndimage, signal


def template_match_opencv(data, template, thresh=0.8,
                          method='cv2.TM_CCOEFF_NORMED'):
    method = eval(method)
    height = template.data.shape[0]
    width = template.data.shape[1]
    points = []
    for i in range(0, data.data.shape[0]):
        result = cv2.matchTemplate(data.data[i, :, :], template.data, method)
        result[result >= thresh] = 1.0
        result[result < thresh] = 0.0
        if result.max() > 0:
            temp = cv2.connectedComponentsWithStats(np.uint8(result),
                                                    connectivity=4)[3][3:]
        for k in range(0, len(temp)):
            points.append([i, temp[k, 0] + width / 2, temp[k, 1] + height / 2])
    points = np.array(points)
    return points


def template_match(data, template, threshold=0.5):
    if len(data.data.shape) != len(template.data.shape):
        raise ValueError('Data shape %s and template shape %s are inconsistent'
                         % (data.data.shape, template.data.shape))

    img = data.data - data.data.mean()
    template.data = template.data - template.data.mean()

    result = signal.fftconvolve(img, template, mode='same')

    threshold = result.max() * threshold
    points = np.float32(peak_local_max(
        result, threshold_abs=threshold)) - [1, 1, -1]
    points_cal = points * data.axes_manager[1].scale

    data.original_metadata.points = points
    data.original_metadata.points_cal = points_cal

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


def plot_all_points(data, axes=['XZ', 'XY']):
    if 'points_cal' in data.original_metadata.keys():
        points_cal = data.original_metadata.points_cal
    else:
        raise ValueError('No points found in metadata')
    maximageXZ = data.max()

    maximageXY = hs.signals.Signal2D(data.max(2))
    maximageXY.axes_manager[0].scale = data.axes_manager[0].scale
    maximageXY.axes_manager[1].scale = data.axes_manager[1].scale

    axes = hs.plot.plot_images([maximageXZ, maximageXY],
                               per_row=2, scalebar='all',
                               colorbar=None, axes_decor='off')
    axes[0].plot(points_cal[:, 2], points_cal[:, 1], 'ro', markersize=2)
    axes[1].plot(points_cal[:, 2], points_cal[:, 0], 'ro', markersize=2)

    return axes


def get_surface(s):
    surface = s.deepcopy()
    surface.change_dtype('float32')
    mid_slice = np.int32(s.data.shape[0] / 2)
    threshold = threshold_otsu(s.data[mid_slice - 10:mid_slice + 10, :, :])
    for i in range(0, surface.data.shape[0]):
        surface.data[i, :, :] = gaussian(surface.data[i, :, :], sigma=[5, 5])
        surface.data[i, :, :][surface.data[i, :, :] < threshold] = 0
        surface.data[i, :, :][surface.data[i, :, :] >= threshold] = 255
        surface.data[i, :, :] = \
            ndimage.binary_fill_holes(surface.data[i, :, :])
        surface.data[i, :, :] = sobel(surface.data[i, :, :])
    return surface


def get_surface3D(data):
    blur = gaussian(data.data, sigma=3)
    thresh_val = threshold_otsu(blur)
    thresholded = blur > thresh_val
    edges = np.zeros_like(thresholded.data)
    for i in range(thresholded.shape[0]):
        edges[i, :, :] = canny(thresholded[i, :, :], sigma=0.1)
    edges = hs.signals.Signal2D(edges)
    return edges


def distance_calc(surface, points):
    mindistance = np.zeros(len(points))
    minloc = np.zeros([len(points), 3])
    surfacepoints = np.array(np.where(surface.data > 0.1)).T
    for i in range(0, len(points)):
        distance = np.sqrt(((surfacepoints - points[i])**2).sum(1))
        mindistance[i] = distance.min()
        minindex = np.argmin(distance)
        minloc[i, :] = surfacepoints[minindex, :]
    return mindistance, minloc
