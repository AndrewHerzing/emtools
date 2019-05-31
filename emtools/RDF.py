import numpy as np
from hyperspy.signals import Signal2D, Signal1D
from hyperspy.drawing.utils import plot_spectra
import matplotlib.pylab as plt


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


def RDF(s, rebinfactor=None, binsize=1):
    if rebinfactor:
        image = s.rebin(scale=(rebinfactor, rebinfactor))
    else:
        image = s.deepcopy()
    fft = image.fft(shift=True)
    psd = Signal2D(np.abs(fft.real())**2,
                   axes=[fft.axes_manager[0].get_axis_dictionary(),
                         fft.axes_manager[1].get_axis_dictionary()])
    xaxis, profile = azimuthalAverage(psd.data, binsize)
    scale = fft.axes_manager[0].scale
    profile = Signal1D(profile)
    profile.axes_manager[0].name = 'Frequency'
    profile.axes_manager[0].units = '1/A'
    profile.axes_manager[0].offset = xaxis[0]*scale
    profile.axes_manager[0].scale = scale
    return psd, profile, xaxis


def plotRDF(rdfs, xrange=None, yrange=None, labels=None):
    if len(rdfs) == 1:
        rdfs.plot()
        ax = plt.gca()
        fig = plt.gcf()
        ax.set_yscale("log")
        ax.set_xscale("log")
        line = ax.get_lines()[0]
        line.set_linewidth(0)
        line.set_marker('o')
        line.set_markeredgecolor('red')
        line.set_markerfacecolor('white')
        if xrange:
            ax.set_xlim(xrange)
        if yrange:
            ax.set_ylim(yrange)
    else:
        ax = plot_spectra(rdfs, legend=labels)
        fig = plt.gcf()
        ax.set_yscale("log")
        ax.set_xscale("log")
        lines = ax.get_lines()
        for line in lines:
            line.set_linewidth(0)
            line.set_marker('o')
            line.set_markerfacecolor('white')
        if xrange:
            ax.set_xlim(xrange)
        if yrange:
            ax.set_ylim(yrange)
    return fig, ax
