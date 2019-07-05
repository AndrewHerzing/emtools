import numpy as np
import cv2
import hyperspy.api as hs

def template_match(data, template, thresh=0.8, method='cv2.TM_CCOEFF_NORMED'):
    method = eval(method)
    height = template.data.shape[0]
    width = template.data.shape[1]
    points = []
    out = data.deepcopy()
    for i in range(0,data.data.shape[0]):
        result = cv2.matchTemplate(data.data[i,:,:],template.data,method)
        result[result>=thresh] = 1.0
        result[result<thresh] = 0.0
        if result.max()>0:
            temp = cv2.connectedComponentsWithStats(np.uint8(result),connectivity=4)[3][3:]
        for k in range(0, len(temp)):
            points.append([i,temp[k,0]+width/2,temp[k,1]+height/2])
    points = np.array(points)
    return points

def plot_points(data, points, index):
    im = data.inav[index].deepcopy()
    for i in np.where(points[:,0]==index):
        print(points[i,1][0])
        m = hs.plot.markers.point(x=points[i,1][0], y=points[i,2][0], color='red')
        im.add_marker(m, permanent=True, plot_marker=False)
    # im.plot()
    # print(im.metadata.Markers)
    return im