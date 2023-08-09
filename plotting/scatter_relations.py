import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage 
from copy import copy
from math import ceil, floor 
from palette import UniStuttgart as uniS


def density_blur( ax, scatters, limits=None, plot_resolution=(800,800), blur_size=(64,64), cmap='jet'  ):
    """
    Map the data onto an arbitrary grid and then create a density blur via convolution
    Maps the data in such a way that each sample is within the given boundary
    Formats the plot based on some default values
    Parameters:
    -----------
    ax:             matplotlib.pyplot.axes object
                    ax object where to put the plot in
    scatters:       numpy nd-array
                    n_samples x 2 data points
    limits:         list of 2 2d-arrays, default None
                    x/ylim for the remapping, if not given defaults to 0, scatters[:,i].max(0)
    plot_resolution:list like of ints
                    resolution of the resulting density plot
    blur_size:      list like of floats
                    approximate size of the blur
    cmap:           str or matplotlib colormap, default 'jet'
                    colormap for plotting, note that jet looks nice but is not scientific
    Returns:
    --------
    blur:           numpy 2d-array
                    image used for plotting of the density blur
    """
    #map data to the grid
    n_samples = scatters.shape[0]
    blur = np.zeros( plot_resolution)
    plot_resolution = np.array( plot_resolution)
    if limits is None:
        limits = [ [0,scatters[:,0].max()],[0,scatters[:,1].max()] ]
    #mapping onto a uniform grid 
    xmap = lambda x: limits[0][0] + plot_resolution[0] * (x-limits[0][0])/limits[0][1] 
    ymap = lambda y: limits[1][0] + plot_resolution[1] * (y-limits[1][0])/limits[1][1] 
    grid_maps = [xmap,ymap]
    for i in range( n_samples):
        idx = [0,0]
        for j in range( 2):
            idx[j] = np.around( grid_maps[j]( scatters[i,j])-1,0 ).astype(int) 
            idx[j] = np.minimum( idx[j], plot_resolution[j]-1 )
        blur[ tuple(idx)] += 1
    blur = np.flip( blur, axis=1).T
    ax.imshow( blur, cmap=cmap, interpolation='none' )
    ## remap the scatters such that you can scatter
    scatter_xmap = lambda x: min( plot_resolution[0]-1, limits[0][0] + plot_resolution[0]/limits[0][1] * x )
    scatter_ymap = lambda x: max( 0, ((limits[1][1]-limits[1][0]) - x/limits[1][1] + 1) * plot_resolution[1] )
    scatter_remap = [scatter_xmap, scatter_ymap]
    scatter_new = np.zeros( scatters.shape) 
    for i in range( n_samples):
        for j in range( 2):
            scatter_new[i,j] = scatter_remap[j]( scatters[i,j] )
    #convolute with gaussian kernel to get the blur
    blur = ndimage.gaussian_filter( blur, blur_size, mode='constant' )
    ## plotting of the data
    x_ticks = 4
    y_ticks = 5
    x_incr = ( limits[0][1]-limits[0][0]) /x_ticks   
    y_incr = ( limits[1][1]-limits[1][0]) /y_ticks   
    xticklabels = np.arange( limits[0][0], limits[0][1] + x_incr, x_incr ) 
    yticklabels = np.arange( limits[1][0], limits[1][1] + y_incr, y_incr )[::-1]
    img =  ax.imshow( blur, cmap=cmap)
    ax.set_xlim( -0.5, blur.shape[1]-0.5 )
    ax.set_ylim( blur.shape[0]-0.5, 0-0.5 )
    ax.set_xticks( np.arange( 0, plot_resolution[0]+1, plot_resolution[0]/x_ticks )-0.5 )
    ax.set_yticks( np.arange( 0, plot_resolution[1]+1, plot_resolution[1]/y_ticks )-0.5)
    ax.set_xticklabels( np.around( xticklabels, 0)[:x_ticks+1] )
    ax.set_yticklabels( np.around( yticklabels, 3)[:y_ticks+1] )
    ## increase the size if we're further away from the bright spot
    base_size = 0.2
    size_increase = 1.8
    bright_spot = np.argwhere( blur == blur.max()).squeeze()[::-1]
    distances = np.linalg.norm( scatter_new - bright_spot, axis=1)**0.5
    #scale distance from 0 to 2
    distances -= distances.min()
    max_distance = np.linalg.norm( plot_resolution- bright_spot )**0.5
    distances = distances/max_distance*2#distances.max() *2 
    sizes = base_size + distances**3 *size_increase 
    ax.scatter( scatter_new[:,0], scatter_new[:,1], color=uniS.lightblue, label='actual values', s=sizes )
    ## mean decorator
    mean_spot = scatter_new.mean(0) 
    mean_spot = [[ *2*[mean_spot[0]], -1e5 ], [1e5, *2*[mean_spot[1]] ] ]
    ax.plot( *mean_spot, ls='--', lw=1.2, markersize=10, color=uniS.blue, marker='x', label='mean' )
    #ax.grid()
    return blur

def compute_sample_bounds( x, y, stepsize=100 ):
    """
    given sampled data y(x), compute the bounds (mean, max and min values) of y(x)
    Gives three arrays of length stepsize which is e.g. min( y(x') ), with x' replicating the given x
    Parameters:
    -----------
    x:          numpy 1darray
                sample input parameters
    y:          numpy 1darray 
                sampled values y(x), length of y and x must match

    stepsize:   int, default 100
                in how many increments the minimum and maximum values should be displayed
    Returns:
    --------
    x':             numpy 1darray
                    virtually sampled x-values for the the computed bounds
    min_bound:      numpy 1darray
                    minimum bounds of the sampled data
    mean_bound:     numpy 1darray
                    mean bounds of the sampled data
    max_bound:      numpy 1darray
                    maximum bounds of the sampled data
    """
    permutation = x.argsort()
    x = x[permutation]
    y = y[permutation]
    x_virt = np.arange( min( x), max(x), (max(x)-min(x))/stepsize  )[:stepsize] #slice again that the array is never too big
    min_bound = np.zeros( stepsize)
    max_bound = np.zeros( stepsize)
    mean = np.zeros( stepsize)
    n = y.shape[-1]
    step = n/stepsize
    for i in range( stepsize):
        min_bound[i] = min(     y[ floor( step*i):ceil( step*(i+1)) ])
        max_bound[i] = max(     y[ floor( step*i):ceil( step*(i+1)) ])
        mean[i]      = np.mean( y[ floor( step*i):ceil( step*(i+1)) ])
    return x_virt, min_bound, max_bound, mean


def voigt_reuss_bounds( kappa_1, kappa_2, volmin=0, volmax=1):
    """
    return the lower or upper hashin shtrikmann bounds for the given
    input parameters kappa. Creates its own discrete sample space for
    the volume fraction on which it is defined
    Paramteres:
    -----------
    kappa_1:    float
                phase parameter of the matrix phase
    kappa_2:    float
                phase parameter of the inclusion phase
    volmin:     float, default 0.
                minimum volume fraction to consider
    volmax:     float, default 0.
                maximum volume fraction to consider
    Returns:
    --------
    vol_virt:   1d-np.array
                sampled volume fraction where lower and upper are defined
    lower:      1d-np.array
                lower hashin-shtrikmann bound 
    upper:      1d-np.array
                upper hashin-shtrikmann bound 
    Example:
    vol_virt, lower, upper = hashin_shtrikman_bounds( 0.2, 1)
    plt.plot( vol_virt, lower)
    plt.plot( vol_virt, upper)
    """
    ## NOTE these are actually voigt reuss bounds
    n=500
    vol = np.arange( volmin, volmax + (volmax-volmin)/n, (volmax-volmin)/n )
    lower = (1-vol)*kappa_1+vol*kappa_2
    upper = 1/( (1-vol)/kappa_1+vol/kappa_2)
    return vol, lower, upper

def voigt_reuss_mech( e_moduli, poisson_ratio, volmin=0, volmax=1, inspected='bulk'):
    """
    return the lower or upper hashin shtrikmann bounds for the given
    input parameters e_moduli. Creates its own discrete sample space for
    the volume fraction on which it is defined
    Paramteres:
    -----------
    e_moduli:       list of two floats
                    E-moduli parameters of matrix, inclusion phase
    possin_ratio:   list of two floats
                    poisson ratio parameters of matrix, inclusion phase
    volmin:         float, default 0.
                    minimum volume fraction to consider
    volmax:         float, default 1.
                    maximum volume fraction to consider
    inspected:      str, default 'bulk'
                    which resposnse to consider given 11 loading
                    defaults to 'bulk', otherwise its 'shear'
    Returns:
    --------
    vol_virt:   1d-np.array
                sampled volume fraction where lower and upper are defined
    lower:      1d-np.array
                lower hashin-shtrikmann bound 
    upper:      1d-np.array
                upper hashin-shtrikmann bound 
    Example:
    vol_virt, lower, upper = hashin_shtrikman_bounds( 0.2, 1)
    plt.plot( vol_virt, lower)
    plt.plot( vol_virt, upper)
    """
    n=500
    vol = np.arange( volmin, volmax + (volmax-volmin)/n, (volmax-volmin)/n )
    e_moduli = np.array( e_moduli ) 
    poisson_ratio = np.array( poisson_ratio ) 
    bulk_modulus = e_moduli / (3*(1-2*poisson_ratio) )
    shear_modulus = e_moduli / (2*(1+poisson_ratio) )
    if inspected == 'bulk':
        max_k = max( e_moduli)
        min_k = min( e_moduli)
        moduli = e_moduli
        #moduli = bulk_modulus
    else:
        max_k = max( poisson_ratio) 
        min_k = min( poisson_ratio) 
        moduli = poisson_ratio 
        #moduli = shear_modulus
    lambd  = (e_moduli*poisson_ratio)/((1+poisson_ratio)*(1-(2*poisson_ratio)))
    mu      = e_moduli/(2*(1+poisson_ratio))
    tl_ones = np.ones( (3,3))
    tl_ones[:,-1] =0
    tl_ones[-1] = 0 
    moduli = []
    for i in range( 2): 
        CC = 2*mu[i] * np.eye(3) + lambd[i]  * tl_ones
        moduli.append( CC )
    ## computation for bulk/shear modulus, simply take the corresponding values
    #upper = 1/( vol/( max_k+ moduli[1] ) + (1- vol)/(max_k + moduli[0] ) ) - max_k
    #lower = 1/( vol/( min_k+ moduli[1] ) + (1- vol)/(min_k + moduli[0] ) ) - min_k
    #lower = (1-vol)*moduli[0]+vol*moduli[1]
    #upper = 1/( (1-vol)/moduli[0]+vol/moduli[1])
    lower = []
    upper = []
    for i,j in [(0,0), (1,1), (1,2), (2,2), (0,2), (1,2) ]:
        lower.append(      (1-vol)*moduli[0][i,j]+vol*moduli[1][i,j] )
        upper.append(  1/( (1-vol)/moduli[0][i,j]+vol/moduli[1][i,j]) )
    return vol, lower, upper




def correlation_matrix( ax, features, targets=None, threshold=None, rearrange=False):
    """
    Return the correlation matrix of the given features showing their
    inbetween correlations. If targets are added it will also display
    the correlations with them.
    Changes the axes object in place, does not have to be caught on return
    TODO can rearange the correlation score based off the cuthill mckee algorithm
    if requested
    also different correlation mb kendalls tau
    Parameters:
    -----------
    ax:         matplotlib.pyplot axes object
                axes handle where to put the plot in
    features:   numpy nd-array
                feature vectors aranged row wise (each row 1 sample)
    targets:    numpy nd-array
                target vectors aranged row wise (each row 1 sample)
    threshold:  float, default None
                at which value the correlations should be white
    Returns:
    --------
    corr_matrix:    numpy nd-array
                    correlation matrix
    graphic:        plt.pyplot mappable
                    mappable to the handle of the imshow object (i think that terminus is correct)
    """
    n_features = features.shape[1]
    if targets is not None:
        n_targets = targets.shape[1]
        matrix = np.abs( np.corrcoef( features, targets, rowvar=False) )
    else:
        matrix = np.abs( np.corrcoef( features, rowvar=False)  )
    print( 'sum of correlations in the kappa rows', matrix[-3:,:-3].sum(1) ) 
    if threshold is not None:
        whites = matrix > threshold 
        matrix[ whites ] = np.nan 
    #cmaps['Diverging'] = [
            #'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            #'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    #cmap = copy( matplotlib.cm.RdBu) 
    cmap = copy( matplotlib.cm.PuOr) 
    cmap.set_bad( 'black') 
    graphic = ax.imshow( matrix, interpolation='nearest', cmap=cmap)
    graphic.set_clim( 0,1)
    #ax.axvline( n_features-0.5, color='black', linewidth=0.8)
    #ax.axhline( n_features-0.5, color='black', linewidth=0.8)
    #ax.axvline( n_features-0.5, color='white', linewidth=0.3)
    #ax.axhline( n_features-0.5, color='white', linewidth=0.3)
    ax.axvline( n_features-0.5, color=uniS.blue, linewidth=1.8)
    ax.axhline( n_features-0.5, color=uniS.blue, linewidth=1.8)
    ax.axvline( n_features-0.5 + n_targets//2, color=uniS.blue, linewidth=1.8, ls='--')
    ax.axhline( n_features-0.5 + n_targets//2, color=uniS.blue, linewidth=1.8, ls='--')
    #ax.set_title( r'pearson correlation matrix\\, black threshold:$R={}r'.format( threshold) )
    #if threshold is not None:
    #    graphic.set_clim( 0,threshold)
    #plt.colorbar( graphic, ax=ax )
    return matrix, graphic

        
def corr_tests( ax, features, targets=None, threshold=None, fun=np.corrcoef):
    n_features = features.shape[1]
    if targets is not None:
        n_targets = targets.shape[1]
        matrix = np.abs( fun( features, targets) )
    else:
        matrix = np.abs( fun( features) )
    if threshold is not None:
        whites = np.abs(matrix) > threshold 
        matrix[ whites ] = np.nan 
    cmap = copy( matplotlib.cm.RdBu) 
    cmap.set_bad( 'black') 
    graphic = ax.imshow( matrix, cmap=cmap)
    ax.axvline( n_features-0.5, color=uniS.blue, linewidth=10)
    ax.axhline( n_features-0.5, color=uniS.blue, linewidth=10)
    #ax.axvline( n_features-0.5, color='white', linewidth=4)
    #ax.axhline( n_features-0.5, color='white', linewidth=4)
    graphic.set_clim( 0,1)
    plt.colorbar( graphic, ax=ax )
    ax.set_title( fun.__name__.replace( '_', ' ') )
    return ax


## other tests using correlation functions
from scipy.spatial.distance import pdist, squareform
def distance_correlation(X, Y):
    """
    Compute the distance correlation function where we do not search
    covariance resemblance but distance resemblence in the data
    Parameters:
    -----------
    X:      numpy nd-array
            first array of samples aranged row wise
    Y:      numpy nd-array
            second array of samples aranged row wise
    Returns:
    --------
    distance_corr:  float or numpy nd-array
                    computed distance correlation
    """
    X = np.array( X) #workaround for h5py
    Y = np.array( Y)
    if Y.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    n_x = X.shape[1]
    n_y = Y.shape[1]
    distance_corr = np.zeros( 2*[n_x+ n_y] )
    ##loop over the features
    for i in range( n_x):
        for j in range( i+1,n_x): 
            distance_corr[i,j] = get_dcor( X[:,i], X[:,j], n)
        for j in range( n_y):
            distance_corr[i,n_x+j] = get_dcor( X[:,i], Y[:,j], n)
    for i in range( n_y):
        for j in range( i+1, n_y):
            distance_corr[n_x+i,n_x+j] = get_dcor( Y[:,i], Y[:,j], n)
    distance_corr += distance_corr.T
    for i in range( n_x+n_y):
        distance_corr[i,i] = 1
    return distance_corr

def get_dcor( x, y, n):
    """ compute the distance correlation of two feature vectors"""
    a = squareform(pdist(x[:,None]))
    b = squareform(pdist(y[:,None]))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    dcov2_xy = (A * B).sum()/(n * n)
    dcov2_xx = (A * A).sum()/(n * n)
    dcov2_yy = (B * B).sum()/(n * n)
    return  np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
