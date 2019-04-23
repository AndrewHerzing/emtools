import numpy as np
import matplotlib
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib as mpl

thermal = matplotlib.colors.LinearSegmentedColormap.from_list('gatan_colormap',['black','blue','green','red','yellow','white'],256,1.0)
JustRed = matplotlib.colors.LinearSegmentedColormap.from_list('red_colormap',['black','red'],256,1.0)
JustGreen = matplotlib.colors.LinearSegmentedColormap.from_list('green_colormap',['black','green'],256,1.0)
JustBlue = matplotlib.colors.LinearSegmentedColormap.from_list('blue_colormap',['black','blue'],256,1.0)

def normalize(image):
        output = image - image.min()
        output = np.uint8(255*output/output.max())
        return(output)
    
def rgboverlay(im1,im2=None,im3=None):
    if len(np.shape(im1))==3:
        rgb = np.dstack((normalize(im1[:,:,0]),normalize(im1[:,:,1]),normalize(im1[:,:,2])))
    else:
        rgb = np.dstack((normalize(im1),normalize(im2),normalize(im3)))
    return(rgb)

def genCMAP(color,alpha=None,N=256):
    if alpha:
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['black',color],N)
        cmap._init()
        cmap._lut[:,-1] = np.linspace(0,alpha,cmap.N+3)
    else:
        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['black',color],N)
    return cmap

def mergeChannels(data,colors=['red','green','blue','cyan','yellow','magenta'],alpha=0.9,limit=256):
    nimages = data.shape[2]
    cmaps = [None]*nimages
    for i in range(0,nimages):
        if i == 0:
            cmaps[i] = genCMAP(colors[i],None,256)
        else:
            cmaps[i] = genCMAP(colors[i],alpha,256)
            
    fig,ax = plt.subplots(1)
    for i in range(0,nimages):
        im = data[:,:,i]
        p2,p98 = np.percentile(im,(1.0,99.950))
        im_rescale = normalize(exposure.rescale_intensity(im,in_range=(p2,p98)))
        _ = ax.imshow(im_rescale, interpolation='nearest', cmap=cmaps[i],clim=[0,limit])
        ax.set_xticks([])
        ax.set_yticks([])
    return fig,ax
