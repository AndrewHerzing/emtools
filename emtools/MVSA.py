import numpy as np
from EMTools import Color
import pylab as plt
from hyperspy.utils.plot import plot_spectra
from hyperspy.drawing.utils import plot_RGB_map as plot_RGB
import hyperspy.api as hspy

def decompositionRGB(data,comps=None,energyrange=None,intensityrange=None,labels=None):
    if comps is None:
        comps = [0,1,2]
    if energyrange is None:
        energyrange = [data.axes_manager[-1].axis[0],data.axes_manager[-1].axis[-1]]
    if intensityrange is None:
        intensityrange = [data.learning_results.factors.min(),data.learning_results.factors.max()]
    fig,ax = plt.subplots(2,1)
    colors=['red','green','blue']
    
    if labels is None:
        algorithm = area1SIbinned.learning_results.decomposition_algorithm.upper()
        labels = [algorithm + ' #%s' % (comps[0]+1),algorithm + ' #%s' % (comps[1]+1),algorithm + ' #%s' % (comps[2]+1)]

    old_shape = [data.learning_results.original_shape[0],
                 data.learning_results.original_shape[1],
                 data.learning_results.loadings.shape[1]]
    
    loadings = data.learning_results.loadings.reshape(old_shape)
    factors = data.learning_results.factors
    images = [None]*loadings.shape[2]
    spectra = [None]*factors.shape[1]
    for i in range(0,len(images)):
        images[i] = hspy.signals.Signal2D(loadings[:,:,i])
        images[i].axes_manager[0].scale = data.axes_manager[0].scale
        images[i].axes_manager[0].offset = data.axes_manager[0].offset
        images[i].axes_manager[0].units = data.axes_manager[0].units
        images[i].axes_manager[1].scale = data.axes_manager[1].scale
        images[i].axes_manager[1].offset = data.axes_manager[1].offset
        images[i].axes_manager[1].units = data.axes_manager[1].units
        spectra[i] = hspy.signals.Signal1D(factors[:,i])
        spectra[i].axes_manager[-1].scale = data.axes_manager[-1].scale
        spectra[i].axes_manager[-1].offset = data.axes_manager[-1].offset
        spectra[i].axes_manager[-1].units = data.axes_manager[-1].units
        spectra[i].axes_manager[-1].name = data.axes_manager[-1].name
        spectra[i].metadata.General.title = 'NMF %s' % str(i+1)
    rgb = plot_RGB([images[comps[0]],images[comps[1]],images[comps[2]]],dont_plot=True)
    plot_spectra([spectra[comps[0]],spectra[comps[1]],spectra[comps[2]]],fig=fig,ax=ax[0],color=['red','green','blue'])
    ax[1].imshow(rgb,interpolation=None)
    ax[1].axis('off')
    ax[0].set_xlim(energyrange[0],energyrange[1])
    ax[0].set_ylim(intensityrange[0],intensityrange[1])
    ax[0].legend(labels)
    fig.tight_layout()
    return fig,ax
	
def varimax(matrix):
    max_iter = 150
    epsilon = 1e-6

    n,k = np.shape(matrix)
    rotm = np.eye(k)
    target_basis = np.eye(n)

    varnow = np.sum(np.var(matrix**2,0))
    not_converged = 1
    iter = 0

    while (not_converged) and (iter<max_iter):
        for j in range(0,k-1):
            for l in range(j+1,k):
                '''Calculate optimal 2-D planar rotation angle for columns j,l'''
                phi_max = np.angle(n*np.sum((matrix[:,j]+1j*matrix[:,l])**4)-np.sum(((matrix[:,j]+1j*matrix[:,l])**2)**2))/4
                sub_rot = [[np.cos(phi_max), -np.sin(phi_max)], [np.sin(phi_max), np.cos(phi_max)]]

                matrix[:,np.ix_([j,l])] = matrix[:,np.ix_([j,l])].dot(sub_rot)
                rotm[:,np.ix_([j,l])] = rotm[:,np.ix_([j,l])].dot(sub_rot)
        varold = varnow
        varnow = np.sum(np.var(matrix**2,0))
        
        if varnow == varold:
            not_converged = 1
        else:
            not_converged = ((varnow-varold)/varnow > epsilon)
        iter+=1
    if iter >= max_iter:
        print('Warning: maximum number of iterations reached')
    else:
        print('Finished after %s iterations' % iter)
    output = target_basis.dot(matrix)
    return(rotm,output)

def PCA(si,normalize=False,newsize=None):
    if si._lazy:
        data = si.deepcopy()
        if newsize:
            data = data.rebin(newsize)
        if normalize:
            data.decomposition(True,output_dimension=64,algorithm='PCA')
        else:
            data.decomposition(False,output_dimension=64,algorithm='svd',centre=None)
        fig,ax = plt.subplots(1)
        ax.set_yscale('log')
        ax.scatter(np.arange(0,30),data.learning_results.explained_variance_ratio[0:30],s=80)
        ax.set_title('Explained Variance Ratio')
        plt.show()
        return(data)
    else:
        data = si.deepcopy()
        if newsize:
            data = data.rebin(newsize)
        if normalize:
            data.decomposition(True,output_dimension=64,algorithm='svd',centre='variables')
        else:
            data.decomposition(False,output_dimension=64,algorithm='svd',centre=None)
        fig,ax = plt.subplots(1)
        ax.set_yscale('log')
        ax.scatter(np.arange(0,30),data.learning_results.explained_variance_ratio[0:30],s=80)
        ax.set_title('Explained Variance Ratio')
        plt.show()
        return(data)

def showResults(model,ncomps=3,shape=None,slice=0):
    print('*******************************************************************************')
    print('SVD Results')
    print('*******************************************************************************')

    if len(model.data.shape) == 2:
        if not shape:
            print('Need to specify shape!')
            return
        nrows,ncols,ntilts = shape
        nchannels = model.data.shape[1]
        loadings = model.learning_results.factors[:,0:ncomps]
        scores = model.learning_results.loadings[:,0:ncomps].reshape(np.append([nrows,ncols,ntilts],ncomps))
        flip = np.sign(loadings.sum(0))
        loadings = loadings*flip
        scores = scores*flip
        for i in range(0,ncomps):
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
            ax2.plot(loadings[:,i])
            ax2.set_title('Spectral Component #%s' % str(i))
            ax1.imshow(scores[:,:,slice,i],cmap='afmhot',interpolation='none')
            ax1.set_title('Spatial Component #%s' % str(i))
        plt.show()
    elif len(model.data.shape) == 3:
        nrows,ncols,nchannels = model.data.shape
        loadings = model.learning_results.factors[:,0:ncomps]
        scores = model.learning_results.loadings[:,0:ncomps].reshape(np.append([nrows,ncols],ncomps))
        for i in range(0,ncomps):
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
            ax2.plot(loadings[:,i])
            ax2.set_title('Spectral Component #%s' % str(i))
            ax1.imshow(scores[:,:,i],cmap='afmhot',interpolation='none')
            ax1.set_title('Spatial Component #%s' % str(i))
        plt.show()
    elif len(model.data.shape) == 4:
        nrows,ncols,ntilts,nchannels = model.data.shape
        loadings = model.learning_results.factors[:,0:ncomps]
        scores = model.learning_results.loadings[:,0:ncomps].reshape(np.append([nrows,ncols,ntilts],ncomps))
        for i in range(0,ncomps):
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
            ax2.plot(loadings[:,i])
            ax2.set_title('Spectral Component #%s' % str(i))
            ax1.imshow(scores[:,:,slice,i],cmap='afmhot',interpolation='none')
            ax1.set_title('Spatial Component #%s' % str(i))
        plt.show()
        
    
def SpatialSimplicity(model,ncomps=None,shape=None,slice=0):
    if len(model.data.shape) == 2:
        print('*******************************************************************************')
        print('Varimax Results')
        print('*******************************************************************************')
        nrows,ncols,ntilts = shape
        nchannels = model.data.shape[1]
        loadings = model.learning_results.factors[:,0:ncomps]
        scores = model.learning_results.loadings[:,0:ncomps].reshape(np.append([nrows,ncols,ntilts],ncomps))
        rot_spec,rotate_spec = varimax(loadings)
        rotate_im = scores.dot(rot_spec)
        flip = np.sign(rotate_spec.sum(0))
        rotate_spec = rotate_spec*flip
        rotate_im = rotate_im*flip
        for i in range(0,ncomps):
                fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
                ax2.plot(rotate_spec[:,i])
                ax2.set_title('Rotated Spectral Component #%s' % str(i))
                ax1.imshow(rotate_im[:,:,slice,i],cmap='afmhot',interpolation='none',clim=[0.0,rotate_im[:,:,slice,i].max()])
                ax1.set_title('Rotated Spatial Component #%s' % str(i))
        plt.show()
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
        rgb = Color.rgboverlay(rotate_im[:,:,slice,:])
        ax1.imshow(rgb,interpolation='none',clim=[0,rgb.max()])
        ax1.set_title('Color Overlay')
        colors=['red','green','blue','cyan','yellow','purple','white','orange','magenta']
        for i in range(0,ncomps):
                ax2.plot(rotate_spec[:,i],color=colors[i])
        ax2.set_title('SS Spectral Components')
        plt.show()
    elif len(model.data.shape) == 3:
        nrows,ncols,nchannels = model.data.shape
        
        print('*******************************************************************************')
        print('Varimax Results')
        print('*******************************************************************************')
        loadings = model.learning_results.factors[:,0:ncomps]
        scores = model.learning_results.loadings[:,0:ncomps].reshape(np.append(np.array(model.data.shape[0:2]),ncomps))
        rot_spec,rotate_spec = varimax(loadings)
        rotate_im = scores.dot(rot_spec)
        flip = np.sign(rotate_spec.sum(0))
        rotate_spec = rotate_spec*flip
        rotate_im = rotate_im*flip
        for i in range(0,ncomps):
                fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
                ax2.plot(rotate_spec[:,i])
                ax2.set_title('Rotated Spectral Component #%s' % str(i))
                ax1.imshow(rotate_im[:,:,i],cmap='afmhot',interpolation='none',clim=[0.0,rotate_im[:,:,i].max()])
                ax1.set_title('Rotated Spatial Component #%s' % str(i))
        plt.show()

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
        rgb = Color.rgboverlay(np.reshape(rotate_im,[nrows,ncols,ncomps]))
        ax1.imshow(rgb,interpolation='none',clim=[200,rgb.max()])
        ax1.set_title('Color Overlay')
        colors=['red','green','blue','cyan','yellow','purple','white','orange','magenta']
        for i in range(0,ncomps):
                ax2.plot(rotate_spec[:,i],color=colors[i])
        ax2.set_title('SS Spectral Components')
        plt.show()
	
    elif len(model.data.shape) == 4:
        ncols,nrows,ntilts,nchannels = model.data.shape
        print('*******************************************************************************')
        print('Varimax Results')
        print('*******************************************************************************')
        loadings = model.learning_results.factors[:,0:ncomps]
        scores = model.learning_results.loadings[:,0:ncomps].reshape(np.append([ncols,nrows,ntilts],ncomps))
        rot_spec,rotate_spec = varimax(loadings)
        rotate_im = scores.dot(rot_spec)
        flip = np.sign(rotate_spec.sum(0))
        rotate_spec = rotate_spec*flip
        rotate_im = rotate_im*flip
        for i in range(0,ncomps):
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
            ax2.plot(rotate_spec[:,i])
            ax2.set_title('Rotated Spectral Component #%s' % str(i))
            ax1.imshow(rotate_im[:,:,slice,i],cmap='afmhot',interpolation='none',clim=[0.0,rotate_im[:,:,slice,i].max()])
            ax1.set_title('Rotated Spatial Component #%s' % str(i))
        plt.show()

        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
        rgb = Color.rgboverlay(np.reshape(rotate_im[:,:,slice,:],[ncols,nrows,ncomps]))
        ax1.imshow(rgb,interpolation='none',clim=[200,rgb.max()])
        ax1.set_title('Color Overlay')
        colors=['red','green','blue','cyan','yellow','purple','white','orange','magenta']
        for i in range(0,ncomps):
            ax2.plot(rotate_spec[:,i],color=colors[i])
        ax2.set_title('SS Spectral Components')
        plt.show()

    output = {}
    output['Specs'] = rotate_spec
    output['Images'] = rotate_im
    return(output)
