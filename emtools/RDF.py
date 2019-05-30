import numpy as np
from scipy import fftpack
import os
import hyperspy.api as hspy


def dataOut(xaxis, data, filename):
    out = np.array([xaxis, data])
    out = out.T
    with open(filename, 'wb') as f:
        np.savetxt(f, out, delimiter=' , ', fmt='%2e')
    f.close()
    return


def processRDF(filename, binsize=1, scalefactor=10, binning=None):
    data = hspy.load(filename)
    data = data.inav[0]
    if binning:
        outshape = [data.data.shape[0]/binning, data.data.shape[1]/binning]
        xaxis, profile = RDF(data.rebin(outshape), binsize, scalefactor)
        return(xaxis, profile)
    xaxis, profile = RDF(data, binsize, scalefactor)
    return xaxis, profile


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


def RDF(image, binsize, scalefactor=None):
    psd = np.abs(fftpack.fftshift(fftpack.fft2(image.data)))**2
    xaxis, profile = azimuthalAverage(psd, binsize)
    if scalefactor:
        scale = scalefactor*image.axes_manager[0].scale
        xaxis = xaxis/(scale*len(psd))
    return xaxis, profile
