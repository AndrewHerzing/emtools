import numpy as np
import cv2

def template_match(data, template, width=4, height=8):
    thresholded = data.deepcopy()
    thresholded.data = np.zeros([HAADF.data.shape[0],173,303],np.uint8)
    points = []
    for i in range(0,data.data.shape[0]):
        result = cv2.matchTemplate(HAADF.data[i,:,:],template,cv2.TM_CCOEFF_NORMED)
        result[result>0.85] = 1
        thresholded.data[i,:,:] = np.uint8(result)
        if (thresholded.data[i,:,:].max()>0):
            temp = cv2.connectedComponentsWithStats(thresholded.data[i,:,:],connectivity=4)[3][1:]+[width,height]
            for k in range(len(temp)):
                points.append(np.append(i,temp[k][::-1]))
    points = np.array(points)
    return points