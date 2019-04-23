import matplotlib
matplotlib.use('Agg')
import hyperspy.api as hspy
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters
from matplotlib_scalebar.scalebar import ScaleBar
import tqdm

recurse = True
rootpath = 'E:/Data/OPV Project/Sebastian/2017-10 FIB Cross Sections'

os.chdir(rootpath)

files = glob.glob(rootpath + '/**/*.ser', recursive=True)

fig,ax = plt.subplots()
for i in tqdm.tqdm(files):
    image = hspy.load(i).inav[0]
    image.data = np.float32(image.data)
    image.data = image.data - image.data.min()
    image.data = np.uint8(255*image.data/image.data.max())
    if i[-5:][0] == '2':
        thresh = filters.threshold_otsu(image.data)
        image.data[image.data>thresh] = thresh     
        outfilename = os.path.splitext(i)[0] + '_HAADF.png'
    else:
        outfilename = os.path.splitext(i)[0] + '_BF.png'
    
    scale = image.axes_manager[0].scale
    units = image.axes_manager[0].units
    
    ax.imshow(image.data)
    scalebar = ScaleBar(dx=scale,units=units,location='lower right',border_pad=0.5)
    ax.add_artist(scalebar)
    fig_size = fig.get_size_inches()
    w,h = image.data.shape
    w2,h2 = fig_size[0],fig_size[1]
    fig.set_size_inches([(w2/w)*w,(w2/w)*h])
    fig.set_dpi((w2/w)*fig.get_dpi())
    fig.patch.set_alpha(0)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    plt.xlim(0,h)
    plt.ylim(w,0)
    fig.savefig(outfilename,overwrite=True,bbox_inches='tight',dpi=100,pad_inches=0,transparent=True)
    plt.cla()
    
plt.close()
