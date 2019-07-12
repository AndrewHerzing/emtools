import numpy as np
import cv2
import matplotlib.pylab as plt
from skimage.feature import match_template, peak_local_max
from skimage.filters import gaussian, threshold_otsu, sobel
from scipy import ndimage


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


def template_match(data, template, threshold=0.7):
    result = match_template(data.data, template.data, pad_input=True)
    points = peak_local_max(result, threshold_abs=threshold)
    return result, points


def plot_points(data, points, index):
    if np.where(np.abs([i[0] - index for i in points]) < 1)[0].any():
        loc = np.where(np.abs([i[0] - index for i in points]) < 1)[0]
        plt.figure()
        plt.imshow(data.inav[index].data)
        for i in range(0, len(loc)):
            plt.scatter(points[loc[i]][2], points[loc[i]][1], s=5,
                        c='red', marker='o', alpha=0.3)
    else:
        nearest = points[np.abs([i[0] - index for i in points]).argmin()][0]
        print('No maxima found at this slice')
        print('Nearest slice with maxima is: %s' % str(nearest))
    return


def get_surface(s):
    surface = s.deepcopy()
    surface.change_dtype('float32')
    mid_slice = np.int32(s.data.shape[0] / 2)
    threshold = threshold_otsu(s.data[mid_slice-10:mid_slice+10, :, :])
    for i in range(0, surface.data.shape[0]):
        surface.data[i, :, :] = gaussian(surface.data[i, :, :], sigma=[5, 5])
        surface.data[i, :, :][surface.data[i, :, :] < threshold] = 0
        surface.data[i, :, :][surface.data[i, :, :] >= threshold] = 255
        surface.data[i, :, :] = \
            ndimage.binary_fill_holes(surface.data[i, :, :])
        surface.data[i, :, :] = sobel(surface.data[i, :, :])
    return surface


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
