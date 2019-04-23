import matplotlib.pylab as plt
from matplotlib_scalebar.scalebar import ScaleBar


def addMarker(image,
              barcolor='black',
              boxcolor='white',
              loc='lower left',
              pad=0.5,
              length=0.2,
              height=0.015):
    scale = image.axes_manager[0].scale
    units = image.axes_manager[0].units
    scalebar = ScaleBar(dx=scale,
                        units=units,
                        height_fraction=height,
                        frameon=True,
                        length_fraction=length,
                        color=barcolor,
                        box_color=boxcolor,
                        location=loc,
                        border_pad=pad)
    plt.gca().add_artist(scalebar)
    return


def show(image,
         figsize=None,
         scalebar=False,
         barcolor='black',
         boxcolor='white'):
    if figsize:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig, ax = plt.subplots()
    ax.imshow(image.data)
    ax.set_yticks([])
    ax.set_xticks([])
    if scalebar:
        addMarker(image, barcolor=barcolor, boxcolor=boxcolor)
    return fig, ax
