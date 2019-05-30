import numpy as np
from hyperspy.signals import Signal2D, Signal1D


def dataOut(xaxis, data, filename):
    out = np.array([xaxis, data])
    out = out.T
    with open(filename, 'wb') as f:
        np.savetxt(f, out, delimiter=' , ', fmt='%2e')
    f.close()
    return


def azimuthalAverage(image, binsize=0.5):
    y, x = np.indices(image.shape)
    center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])
    nbins = int(np.round(r.max() / binsize)+1)
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins+1)
    bin_centers = (bins[1:]+bins[:-1])/2.0

    radial_prof = np.histogram(r, bins, weights=(image))[0] / \
        np.histogram(r, bins, weights=np.ones(np.shape(image)))[0]
    return bin_centers, radial_prof


def RDF(s, rebinfactor=None, binsize=1, scalefactor=10):
    if rebinfactor:
        image = s.rebin(scale=(rebinfactor, rebinfactor))
    else:
        image = s.deepcopy()
    fft = image.fft(shift=True)
    psd = Signal2D(np.abs(fft.real())**2,
                   axes=[fft.axes_manager[0].get_axis_dictionary(),
                         fft.axes_manager[1].get_axis_dictionary()])
    xaxis, profile = azimuthalAverage(psd.data, binsize)
    profile = Signal1D(profile)
    if scalefactor:
        scale = scalefactor*image.axes_manager[0].scale
        xaxis = xaxis/(scale*len(psd))
    return xaxis, profile
