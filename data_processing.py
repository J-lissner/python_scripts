import numpy as np
import h5py
from math import ceil, floor

##################### GENERAL DATA TRANSFORMATION FOR ARBITRARY DATA ####################
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
    x_train, y_train, x_valid, y_valid:     nd-numpy arrays
                Input (x) and output (y) values as a training and validation set
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
        inputs = inputs[shuffle]
        outputs = outputs[shuffle]
    return inputs[:n_train], outputs[:n_train], inputs[n_train:], outputs[n_train:]


def filter_samples( x, y, xmin=None, xmax=None, ymin=None, ymax=None ):
    """
    Find the samples which fullfill the condition in x and y.
    select the bound of values 
    min finds values larger than specified, max finds values smaller than specified
    Returns the indices of the values which fullfill the shared condition/bound
    Example:
    --------
    Imagine a scatterplot, then with x/ymin,x/ymax rectangular regions are specified
    Indices of the samples in this rectangular region are returned.
    Parameters:
    -----------
    x:      numpy 1d-array
            array of first entires
    y:      numpy 1d-array
            array of second entires
    xmin:   float, default None
            lower bound of the region of x
    xmax:   float, default None
            upper bound of the region of x
    ymin:   float, default None
            lower bound of the region of y
    ymax:   float, default None
            upper bound of the region of y
    Returns:
    --------
    indices:    numpy 1d-array
                found indices which fullfill these condition 
    """
    if xmin is None and xmax is None:
        print( 'please specify at least one x-bound, returning empty set' )
        return np.array([])
    if ymin is None and ymax is None:
        print( 'please specify at least one y-bound, returning empty set' )
        return np.array([])
    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()
    if ymin is None:
        ymin = y.min()
    if ymax is None:
        ymax = y.max()
    legit_x = ( xmin < x) * ( x < xmax)
    legit_y = ( ymin < y) * ( y < ymax)
    return np.argwhere( legit_x*legit_y).squeeze() 


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
        print( '########## Error Message ##########\nno valid scaling specified, returning unscaled data and no scaling')
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
    if scaling[2] in [ 'single_std1', 'combined_std1']:
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
    if shift[2] in [ 'single_std1', 'combined_std1']:
        data = data / shift[1] + shift[0] 
    elif shift[2] == '0-1':
        data = data * shift[1] + shift[0] 
    elif shift[2] == '-1,1':
        data = (data + 1) * shift[1]/2 + shift[0]  
    else:
        print('No valid scaletype given, returning raw data' )
    return data

def batch_data( x, y, n_batches, shuffle=True, stochastic=0.0, factory=False):
    """
    Generator/Factory function, yields 'n_batches' batches when called as
    a 'for loop' argument.  The last batch is the largest if the number of
    samples is not integer divisible by 'n_batches' (the last batch is at
    most 'n_batches-1' larger than the other batches)
    Also enables a stochastic chosing of the training samples by ommiting
    different random samples each epoch
    Parameters:
    -----------
    x:              numpy array
                    input data aranged column wise
    y:              numpy array
                    output data/target values aranged column wise
    n_batches:      int
                    number of batches to return
    shuffle:        bool, default True
                    If the data should be shuffled before batching
    stochastic:     float, default 0.5
                    if the data should be stochastically picked, has to be <=1
                    only available if <shuffle> is True
    factory:        bool, default False
                    if this function should work as a generator function
    Yields:
    -------
    x_batch         numpy array
                    batched input data
    y_batch         numpy array
                    batched output data
    """
    n_samples = y.shape[0]
    if shuffle:
        permutation = np.random.permutation( n_samples )
        x = x[ permutation]
        y = y[ permutation]
    else:
        stochastic = 0
    batchsize = int( n_samples // n_batches * (1-stochastic) )
    max_sample = int( n_samples* (1-stochastic) )
    i = -1 #to catch errors for n_batches == 1
    if factory:
        for i in range( n_batches-1):
            yield x[i*batchsize:(i+1)*batchsize], y[i*batchsize:(i+1)*batchsize]
        yield x[(i+1)*batchsize:max_sample ], y[(i+1)*batchsize:max_sample ]
    else:
        batches = []
        for i in range( n_batches-1):
            batches.append( x[..., i*batchsize:(i+1)*batchsize], y[..., i*batchsize:(i+1)*batchsize] )
        batches.append( x[..., (i+1)*batchsize:max_sample ], y[..., (i+1)*batchsize:max_sample ] )
        return batches



def compute_error( true_value, predictions, scaling=None, convertScale=False, metric='mse'):
    """
    Compute the error between true value and predictions based on 
    specified metric with/without scaling back to the real scale 
    INPUT(s):
        true_value      : array of scaled true_value
        predictions     : array of scaled predictions (or prediction means for bayesian NN)
        convertScale    : boolean indicating whether the values are to be scaled to the real scale
        scaling         : list with scaling info from data_processing.scale_data()
        metric          : metric to be used to compute the error
                            (a) 'mse' : Mean Squared Error
                            (b)
                            (c)
    OUTPUT(s):
        error           : scalar error value
    # TODO rewrite a few things here
    """
    # Scaling back to the real scale
    if convertScale:
        true_value  = get.unscale_data(true_value,  scaling)
        predictions = get.unscale_data(predictions, scaling)
    # Computing error based on the metric
    if metric=='mse':
        error = np.square(np.subtract(true_value, predictions)).mean() 
    return error

