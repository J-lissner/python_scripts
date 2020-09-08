import numpy as np
import h5py
from numpy.fft import fftn, ifftn
from math import ceil, floor

#####################################AQUISITION AND TRANSFORMATION OF BINARY IMAGE DATA ###############################

def load_datasets( n_snapshots, dataset_name='dset_{}', dataset_counter=0, specified_indices=None,  filename=None, hdf5_path='image_data'):
    """
    Load and vectorize multiple 3d RVE snapshots stored in a hdf5 file
    Puts the vectorized RVE into a column array (each column one RVE) and returns an array of size "resolution x n_snapshots"
    It is assumed that each RVE is its own dataset, and that they are in the same folder with the same name, except a numbering index
    Parameters:
    -----------
    n_snapshots:        int
                        number of vectorized snapshots to return

    dataset_name:       parsing string, default "dset_{}"
                        name of the RVE datasets in the hdf5 file
    dataset_counter:    int, default 0
                        which RVE to return, returns the RVE "dset_0" until "dset_'n_snapshots'"
    specified_indices:  list, default None
                        If a list is given, "dataset_counter" is ommited and it only returns the RVE wit the requested number
    filename:           string, default None
                        path and name of the hdf5 file, defaults to "my dataverse" on emmawork2 (JL)
    hdf5_path:          string, default "image_data"
                        Internal path to the snapshot data in the hdf5 file

    Returns:
    --------
    snapshots:          numpy ndarray
                        loaded and vectorized RVE with each column one RVE (still returns a 2d-array on 1 RVE)
    """
    if filename is None:
        filename = '/scratch/lissner/dataverse/3d_rve.h5'
    h5file = h5py.File( filename, 'r')
    image_location = h5file[ hdf5_path ]

    snapshots = []
    if not specified_indices is None: #return the images with the specified index
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
#######################################################################################################################
######################################### GENERAL DATA TRANSFORMATION FOR ARBITRARY DATA ##############################

def split_data( inputs, outputs, split=0.3, shuffle=True):
    """ 
    Randomly shuffle the data and thereafter split it into two sets 
    Arranges the data row-wise if it is given column wise (return arrays, each row one sample)
    (Note that this function assumes that there are more samples than dimensions of the problem)
    Parameters:
    -----------
    inputs:     numpy nd-array
                input data (preferably) arranged row wise
    outputs:    numpy nd-array
                output data (preferably) arranged row wise
    split:      float, default 0.3
                percentage part of the second set (validation set)
    shuffle:    bool, default True
                Whether the data should be randomly shuffled before splitting
    Returns:
    --------
    input_train:    numpy nd-array
                    input data containing 1-split percent of samples (rounded up)
    input_valid:    numpy nd-array
                    input data containing split percent of samples (rounded down)
    output_train:   numpy nd-array
                    output data containing 1-split percent of samples (rounded up)
    output_valid:   numpy nd-array
                    output data containing split percent of samples (rounded down) 
    """
    # It is assumed that there is more data provided than dimension of the input/output 
    # Hence, transpose the data if it is arranged "column wise"
    if inputs.shape[0] < inputs.shape[1]:
        print( 'Transposing inputs before splitting and shuffling such that each row is one data-sample')
        print( '...returning row wise aranged inputs')
        inputs = inputs.T
    if outputs.shape[0] < outputs.shape[1]:
        print( 'Transposing outputs before splitting and shuffling such that each row is one data-sample')
        print( '...returning row wise aranged outputs')
        outputs = outputs.T
    n_data  = inputs.shape[0]
    n_train = ceil( (1-split) * n_data )
    if shuffle is True:
        shuffle = np.random.permutation(n_data)
        x_train = inputs[ shuffle,:][:n_train,:]
        x_valid = inputs[ shuffle,:][n_train:,:]
        y_train = outputs[ shuffle,:][:n_train,:]
        y_valid = outputs[ shuffle,:][n_train:,:]
        return x_train, y_train, x_valid, y_valid
    else:
        return inputs[:n_train,:], outputs[:n_train,:], inputs[n_train:,:], outputs[n_train:,:]

def scale_data( data, slave_data=None, scaletype='single_std1'):
    """
    Compute a shift based on data and apply it to slave_data
    Data has to be arranged row wise (each row one sample)
    Choose between the following scale methods:
    'single_std1': scale each component over all samples to have 0 mean and standard devaition 1
    'combined_std1': scale all component (combined) over all samples to have 0 mean and standard devaition 1
    '0-1': scale each component to lie on the interval [0,1]
    '-1-1': scale each component to lie on the interval [-1,1]
    Parameters:
    -----------
    data:   numpy nd-array
            data to compute the shift/scaling
    slave_data:     numpy nd-array, default None
                    data to apply the shift to
    scaletype:      string, default 'single_std1'
                    specified scaletype to compute/apply
    Returns:
    --------
    data:       numpy nd-array
                shifted and scaled data
    slave_data: numpy nd-array, or None
                shifted and scaled slave data or None if no slave data was given
    scaling:    list of three
                Parameters and type of the scaling ( can be used in other functions)
    """
    if data.shape[0] < data.shape[1]:
        print( 'Transposing inputs such that each row is one data-sample, assuming that given "slave_data" is of the same format')
        print( '...returning row wise aranged data')
        data = data.T
        if slave_data:
            slave_data = slave_data.T
    n, m  = data.shape
    scaling = [None,None, scaletype]
    if scaling[2] == 'single_std1':
        scaling[0] = np.mean( data, 0)
        data       = data - scaling[0]
        scaling[1] = np.sqrt( n-1) / np.sqrt( np.sum( data**2, 0)) 
        data       = scaling[1] * data

    elif scaling[2] == 'combined_std1':
        scaling[0] = np.mean( data,0) 
        data       = data - scaling[0]
        scaling[1] = np.sqrt( m*n-1) / np.linalg.norm( data,'fro') 
        data       = scaling[1] * data

    elif scaling[2] == '0-1':
        scaling[0] = np.min( data, 0)
        data       = data - scaling[0]
        scaling[1] = np.max( data, 0)
        data       = data /scaling[1]

    elif scaling[2] == '-1,1':  
        scaling[0] = np.min( data, 0)
        data       = data - scaling[0]      
        scaling[1] = np.max( data, 0)     
        data       = data /scaling[1] *2 -1 

    else: 
        print( '########## Error Message ##########\nno valid scaling specified, returning unscalinged data and no scaling')
        print( "valid options are: 'single_std1', 'combined_std1', '0-1', '-1,1', try help(scale_data)\n###################################" )
    if slave_data is not None:
        slave_data = scale_with_shifts( slave_data, scaling)
        return data, slave_data, scaling #automaticall returns the unscalinged data if a wrong scaling is specified
    else:
        return data, scaling


def scale_with_shifts( data, scaling):
    """
    Apply the known shift computed with 'scale_data()' to some data
    Parameters:
    -----------
    data:       numpy nd-array
                data arranged row wise
    scaling:    list of three
                previously computed scaling from 'scale_data()'
    Returns:
    --------
    data:       numpy nd-array
                scaled data using 'scaling' 
    """
    if scaling[2] == 'single_std1' or scaling[2] == 'combined_std1':
        data = scaling[1] * ( data - scaling[0] ) 
    elif scaling[2] == '0-1':
        data = (data - scaling[0]) /scaling[1] 
    elif scaling[2] == '-1,1':
        data = (data - scaling[0]) /scaling[1] *2 -1 
    else: 
        print('Invalid scaletype given, returning raw data')
    return data


def unscale_data( data, shift ):
    """
    Unscale the given data based on the know shift computed from 'scale_data()'
    Parameters:
    -----------
    data:       numpy nd-array
                scaled data arranged row wise
    scaling:    list of three
                previously computed scaling from 'scale_data()'
    Returns:
    --------
    data:       numpy nd-array
                unscaled data using 'scaling' 
    """
    if shift[2] == 'single_std1' or shift[2] == 'combined_std1':
        data = data / shift[1] + shift[0] 
    elif shift[2] == '0-1':
        data = data * shift[1] + shift[0] 
    elif shift[2] == '-1,1':
        data = (data + 1) * shift[1]/2 + shift[0]  
    else:
        print('No valid scaletype given, returning raw data' )
    return data


def batch_data( x, y, n_batches, shuffle=True):
    """
    Take input and output data and put them into 'n_batch' batches
    Returns a tuple of the form ( (in_1, out_1), (in_2, out_2), ...)
    If the number of samples is not divisible by "n_batch", the last batch will be the largest
    (but at most n_batches-1 larger than the other batches)
    Paramteres:
    -----------
    x:          numpy nd array
                input data, arranged row wise (each row one sample)
    y:          numpy nd array
                output data corresponding to the input data (also row wise)
    n_batch:    int
                number of batches there should be returned 
    shuffle:    bool, default True
                if the data should be shuffled before batching it
    Returns:
    --------
    batched_data:   tuple
                    tuple of batched data in the form ( (x_1, y_1), (x_2, y_2), ...)
    """
    n_samples = y.shape[0]
    if shuffle:
        permutation = np.random.permutation( n_samples )
        x = x[permutation]
        y = y[permutation]
    batchsize = floor( n_samples/ n_batches)
    i = -1 # set a value that for n_batches=1 it does return the whole set
    batches = []
    for i in range( n_batches-1):
        batches.append( (x[ i*batchsize:(i+1)*batchsize], y[ i*batchsize:(i+1)*batchsize]) )
    batches.append( (x[(i+1)*batchsize:], y[(i+1)*batchsize:]) )
    return batches


####################################################################################################
### DEPECRATED BUT LEFT temporarily (only works for the 2d file with very specific formatting
### The function has been replaced by the "load_datasets" function for a more general saving format
####################################################################################################
def load_snapshots( n, dset_nr=1, data_file=None, memory_efficient=False):
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
        circle_index     = np.sort( np.random.permutation( size_dataset)[:n_circle] ) #to access hdf5 files the indices need to be sorted
        rectangle_index  = np.sort( np.random.permutation( size_dataset)[:n_rectangle] ) #These statements are only useful if n/2 <= size_dataset
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


