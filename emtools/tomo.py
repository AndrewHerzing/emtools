import numpy as np
import cv2
import matplotlib.pylab as plt
from skimage.feature import match_template, peak_local_max
from skimage.filters import gaussian, threshold_otsu, sobel


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
    # result = data.deepcopy()
    result = match_template(data.data, template.data, pad_input=True)
    points = peak_local_max(result, threshold_abs=0.5)
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


def get_surface(data):
    surface = data.deepcopy()
    surface.data = gaussian(surface.data, sigma=[3, 3, 3])

    threshold = threshold_otsu(surface.data)
    surface.data[surface.data < threshold] = 0
    surface.data[surface.data >= threshold] = 255
    for i in range(0, surface.data.shape[0]):
        surface.data[i, :, :] = sobel(surface.data[i, :, :])
    return surface
