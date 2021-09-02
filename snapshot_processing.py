import numpy as np
import h5py
from numpy.fft import fftn, ifftn
from math import ceil, floor


#####################################AQUISITION AND TRANSFORMATION OF BINARY IMAGE DATA ###############################

def load_datasets( n_snapshots, dataset_counter=0, filename=None, **default_kwargs):
    """
    Load and vectorize multiple 3d RVE snapshots stored in a hdf5 file
    Puts the vectorized RVE into a column array (each column one RVE) and returns an array of size "resolution x n_snapshots"
    It is assumed that each RVE is its own dataset, and that they are in the same folder with the same name, except a numbering index
    Parameters:
    -----------
    n_snapshots:        int
                        number of vectorized snapshots to return 
    dataset_counter:    int, default 0
                        which RVE to return, returns the RVE "dset_0" until "dset_'n_snapshots'"
    filename:           string, default None
                        path and name of the hdf5 file, defaults to "my dataverse" on emmawork2 (JL)

    **default_kwargs:   keyworded arguments with default values
                        The remaining values below denote the defaults. 
    dataset_name:       parsing string, default "dset_{}"
                        name of the RVE datasets in the hdf5 file
    specified_indices:  list, default None
                        If a list is given, "dataset_counter" is ommited and it only returns the RVE wit the requested number
    hdf5_path:          string, default "image_data"
                        Internal path to the snapshot data in the hdf5 file

    Returns:
    --------
    snapshots:          numpy ndarray
                        loaded and vectorized RVE with each column one RVE (still returns a 2d-array on 1 RVE)
    """
    if filename is None:
        filename = '/scratch/lissner/dataverse/3d_rve.h5' 
    dataset_name = default_kwargs.pop( dataset_name, 'dset_{}' )
    specified_indices = default_kwargs.pop( specified_indices, None )
    hdf5_path = default_kwargs.pop( hdf5_path, 'image_data' )
    if default_kwargs:
        print( 'non specified default_kwargs given in "load_datasets", those are', default_kwargs.keys() )
        print( 'continuing program, unexpected behaviour may occur!' )
    
    h5file = h5py.File( filename, 'r')
    image_location = h5file[ hdf5_path ]
    snapshots = []
    if specified_indices is not None: #return the images with the specified index
        for index in specified_indices:
            dataset = dataset_name.format( index)
            snapshots.append( image_location[ dataset][:].flatten() )
    else: # take the first n consecutive RVE
        for i in range( n_snapshots):
            dataset = dataset_name.format( dataset_counter +i)
            snapshots.append( image_location[ dataset][:].flatten() )
    h5file.close()
    if snapshots[0].ndim != 1:
        return snapshots
    else:
        return np.vstack( snapshots).T


def correlation_function( images, fourier=False, resolution=None):
    """
    compute the spatial correlation function (2PCF) the given snapshots
    It can be chosen if the 2pcf in the fourier spectrum is returned, or the full computation is done
    The input and output data is arranged column wise (each column one image)
    NOTE that some functions below use data aquired from this function, only works for binary image data
    Parameters:
    -----------
    images      nd-numpy array
                vectorized image data of multiple images
                It is assumed that there are fewer images than voxel per image
    fourier:    bool or list/numpy 1d-array, default False
                If true, does not compute the IFFT and returns the fourier spectrum
                If an array like is given, "fourier" is used to truncate (via indexing) for efficient computation
    Resolution: list like of integers, default None
                Specificy the resolution of each image if they are not of square resolution
    Returns:
    --------
    2pcf:       numpy nd-array
                computed two point correlation function for each image
                if more than one image was given they are arranged column wise (each column one pcf) 
    """
    ## preallocation and default parameters
    dim, n_s = images.shape
    if dim < n_s:
        print( 'Transposing the images to arrange them column wise (non tensorflow conform)' )
        dim, n_s = n_s, dim
        images = images.T
    if not (fourier is True or fourier is False):
        dim = len( np.nonzero( np.array( fourier) )[0] )
    if resolution==None:
        square_size= dim**0.5
        error_msg  = 'No square resolution detected, please specify original resolution of the images in tuple format'
        assert square_size == round(square_size), error_msg
        resolution = ( int(square_size), int(square_size) )

    ## computation of the 2pcf
    c11 = np.zeros( (dim, n_s), dtype=float )
    scaling = np.prod( resolution)
    for i in range(n_s):
        c1 = fftn( images[:,i].reshape( resolution))
        if fourier is False:
            c1 = np.conj(c1) * c1 / scaling
            c1 = ifftn(c1.real)
        elif fourier is True:
            c1 = np.conj(c1) * c1 / scaling #(scaling**2)
        else:  #Computation of truncated 2pcf
            c1 = c1.flatten()[ fourier]
            c1 = np.conj( c1) * c1 / scaling #(scaling**2)
        c11[:,i] = (c1.real ).flatten()
    print( 'DISCLAIMER REALLY IMPORTANT\n The 2pcf in fourier is buggy, but everything has been computed that way, FOR THE NEXT STEP RETRAIN THE ANN (because the scaling was off (correct scaling commented out)) RB is normed -> no retraining required' )
    return c11


def process_snapshots( snapshots, fourier_space=False, scaletype='fscale'):
    """
    Scale snapshots of the 2 point correlation function, multiple snapshots arranged column wise
    Linear scaling, always sets a zero mean. Choose between the following methods:
    None: pcf - mean( pcf)
    'fscale': (pcf - mean( pcf) )/ vol      vol = volume fraction
    'max1': (pcf - mean( pcf) ) / ( vol - vol**2 )
    Parameters:
    -----------
    snapshots:      numpy nd-array
                    snapshots of the 2pcf arranged column wise
    fourier_space:  bool, default False
                    Whether 'fourier snapshots' are given
    scaletype:      string, default 'fscale'
                    Scaletype to choose of, see documentation above
    Returns:
    --------
    snapshots:      numpy nd-array
                    scaled snapshots arranged column wise (each column one sample)
    vol_2:          float
                    volume fraction of the inclusion phase 
    """
    vol_2 = snapshots[0,:] 
    if fourier_space is True:
        vol_2          = np.sqrt( vol_2 )
        snapshots[0,:] = 0 #zero mean snapshot
    else:
        snapshots = snapshots - vol_2**2 #zero mean
    if scaletype == 'max1': #every corner has value 1
        snapshots = snapshots/ ( vol_2 -vol_2**2 )
    elif scaletype == 'fscale': #take out the volume fraction
        snapshots = snapshots / vol_2  #"f_scale"
    return snapshots, vol_2


def reduced_coefficients( basis, snapshots, fourier_truncation=None, dim=30):
    """
    Compute the reduced coefficients of the snapshots with the reduced basis
    Choose between a normal or a truncated fourier basis for the computation
    Parameters:
    -----------
    basis:                  numpy nd-array
                            reduced basis with eigenmodes arranged column wise 
    snapshots:              numpy nd-array
                            snapshots of the data arranged column wise
    fourier_truncation:     indexing array, default None
                            If a fourier basis is handed, this input denotes
                            the truncation array for each eigenmode
    dim:                    int, default 30
                            number of reduced coefficients to compute
    Returns:
    --------
    xi:     numpy nd-array
            reduced coefficients for each snapshots arranged column wise
    """
    if basis.shape[1]< dim:
        print("{} eigenmodes don't existent, taking only the first {} existing ones".format( dim, basis.shape[1]) )
        dim = basis.shape[1]
    if isinstance( fourier_truncation, np.ndarray) or isinstance( fourier_truncation, list):
        return basis[:,:dim].T @ snapshots[fourier_truncation]
    else:
        return basis[:,:dim].T @ snapshots

# this seems so much nicer with a class for snapshots, correlation function class or so, because it is problem specific 

####################################################################################################
### DEPECRATED BUT LEFT temporarily (only works for the 2d file with very specific formatting
### The function has been replaced by the "load_datasets" function for a more general saving format
####################################################################################################
def load_snapshots( n, dset_nr=1, data_file=None, memory_efficient=False, draw_random=True):
    """
    load in the snapshots, not really an elegant solution
    returns approximately n/2 random snapshots of each class (circle and rectangular)
    chosen from dataset with the given dset_nr
    This one needs to be revorked, it is better to load the whole datasets and then permute it later (in the loop)
    """
    if data_file is None:
        data_file = '/scratch/lissner/projekte/pred_kappa/snapshot_data/all_30000_images.hdf5'
    with h5py.File( data_file, 'r' ) as raw_data:
        n_circle         = ceil( n/2)
        n_rectangle      = floor( n/2)
        size_dataset     = raw_data['circle/image_data/dset_{}'.format( dset_nr) ].shape[1]
        if draw_random is True:
            circle_index     = np.sort( np.random.permutation( size_dataset)[:n_circle] ) #to access hdf5 files the indices need to be sorted
            rectangle_index  = np.sort( np.random.permutation( size_dataset)[:n_rectangle] ) #These statements are only useful if n/2 <= size_dataset
        else:
            circle_index = slice( 0, n_circle)
            rectangle_index = slice( 0, n_rectangle)
        if memory_efficient:
            circle_images    = raw_data['circle/image_data/dset_{}'.format( dset_nr) ][:,circle_index]
            rectangle_images = raw_data['rectangle/image_data/dset_{}'.format( dset_nr) ][:,rectangle_index]
        else:
            circle_images    = raw_data['circle/image_data/dset_{}'.format( dset_nr) ][:]
            circle_images    = circle_images[:, circle_index]
            rectangle_images = raw_data['rectangle/image_data/dset_{}'.format( dset_nr) ][:]
            rectangle_images = rectangle_images[:, rectangle_index] 
        images           = np.hstack(( circle_images, rectangle_images))
        
        circle_target    = raw_data['circle/target_values/heat_conduction/dset_{}'.format( dset_nr)][:,circle_index]
        rectangle_target = raw_data['rectangle/target_values/heat_conduction/dset_{}'.format( dset_nr)][:,rectangle_index]
        target           = np.hstack(( circle_target, rectangle_target))

    return images, target


