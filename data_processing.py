import numpy as np
import sys
from math import ceil, floor
try: import tensorflow as tf
except: pass

##################### GENERAL DATA TRANSFORMATION FOR ARBITRARY DATA ####################
def split_data( inputs, outputs, split=0.3, shuffle=True, slave_inputs=None, slave_outputs=None):
    """ 
    Randomly shuffle the data and thereafter split it into two sets 
    Arranges the data row-wise if it is given column wise (return arrays, each row one sample)
    (Note that this function assumes that there are more samples than dimensions of the problem)
    Parameters:
    -----------
    inputs:         numpy nd-array
                    input data (preferably) arranged row wise
    outputs:        numpy nd-array
                    output data (preferably) arranged row wise
    split:          float, default 0.3
                    percentage part of the second set (validation set)
    shuffle:        bool, default True
                    randomly shuffly the data before splitting
    slave_inputs:   numpy nd-array, default None
                    Any dependend input data which should be split analogously
    slave_outputs:  numpy nd-array, default None
                    Any dependend output data which should be split analogously
    Returns:
    --------
    x_train, y_train, x_valid, y_valid:     nd-numpy arrays
                Input (x) and output (y) values as a training and validation set
    """
    # It is assumed that there is more data provided than dimension of the input/output 
    # Hence, transpose the data if it is arranged "column wise"
    if inputs.shape[0] < inputs.shape[1] and inputs.ndim < 3:
        print( 'Transposing inputs before splitting and shuffling such that each row is one data-sample')
        print( '...returning row wise aranged inputs')
        inputs = inputs.T
    if outputs.shape[0] < outputs.shape[1] and outputs.ndim < 3:
        print( 'Transposing outputs before splitting and shuffling such that each row is one data-sample')
        print( '...returning row wise aranged outputs')
        outputs = outputs.T
    n_data  = inputs.shape[0]
    n_train = ceil( (1-split) * n_data )
    if shuffle is True:
        shuffle = np.random.permutation(n_data)
        inputs = inputs[shuffle]
        outputs = outputs[shuffle]
        slave_inputs = slave_inputs[shuffle] if slave_inputs is not None else slave_inputs
        slave_outputs = slave_outputs[shuffle] if slave_outputs is not None else slave_outputs
    ## conditional return statements depending on the amount of sets
    if slave_inputs is None and slave_outputs is None: 
        return inputs[:n_train], outputs[:n_train], inputs[n_train:], outputs[n_train:]
    elif slave_outputs is None:
        return inputs[:n_train], outputs[:n_train], inputs[n_train:], outputs[n_train:], slave_inputs[:n_train], slave_inputs[n_train:]
    elif slave_inputs is None:
        return inputs[:n_train], outputs[:n_train], inputs[n_train:], outputs[n_train:], slave_outputs[:n_train], slave_outputs[n_train:]
    else:
        return inputs[:n_train], outputs[:n_train], inputs[n_train:], outputs[n_train:], slave_inputs[:n_train], slave_inputs[n_train:], slave_outputs[:n_train], slave_outputs[n_train:]



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
    'single_std1'/'default': scale each component over all samples to have 0 mean and standard devaition 1
    'combined_std1': scale all component (combined) over all samples to have 0 mean and standard devaition 1
    'covariance_shift': scale by 
    '0,1': scale each component to lie on the interval [0,1]
    '-1,1': scale each component to lie on the interval [-1,1]
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
    if data.shape[0] < data.shape[1] and data.ndim < 3:
        print( 'Transposing inputs such that each row is one data-sample, assuming that given "slave_data" is of the same format')
        print( '...returning row wise aranged data')
        data = data.T
        if slave_data is not None:
            slave_data = slave_data.T
    n, m  = data.shape[:2]
    axis = tuple( range( data.ndim-1) )
    scaling = [None,None, scaletype]
    if scaling[2] is None:
        pass
    elif scaling[2].lower() in [ 'default', 'single_std1'] :
        scaling[0] = np.mean( data, axis=axis)
        data       = data - scaling[0]
        scaling[1] = np.sqrt( (n-1) / np.sum( data**2, axis=axis) ) 
        data       = scaling[1] * data

    elif scaling[2] == 'combined_std1':
        scaling[0] = np.mean( data,axis=axis) 
        data       = data - scaling[0]
        scaling[1] = np.sqrt( m*n-1) / np.linalg.norm( data,'fro') 
        data       = scaling[1] * data

    elif 'cov' in scaling[2].lower(): 
        nugget = 1e-12
        scaling[0] = data.mean( 0 )
        data = data - scaling[0]
        scaling[1] = np.cov( data, rowvar=False)  + nugget * np.eye(data.shape[1] )
        #scaling[1] = np.linalg.cholesky( np.cov( data, rowvar=False)+ nugget * np.eye(data.shape[1] )) 
        data =  data @ np.linalg.inv( scaling[1] ) 

    elif '0' in scaling[2] and '1' in scaling[2]:
        scaling[0] = np.min( data, axis=axis)
        data       = data - scaling[0]
        scaling[1] = np.max( data, axis=axis)
        data       = data /scaling[1]

    elif '-1' in scaling[2] and '1' in scaling[2]:
        scaling[0] = np.min( data, axis=axis)
        data       = data - scaling[0]      
        scaling[1] = np.max( data, axis=axis)     
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
    if scaling[2] is None:
        pass
    elif scaling[2] in [ 'single_std1', 'combined_std1', 'default']:
        data = scaling[1] * ( data - scaling[0] ) 
    elif  'cov' in scaling[2].lower():
        data = (data-scaling[0]) @ np.linalg.inv( scaling[1])
    elif '0' in scaling[2] and '1' in scaling[2]:
        data = (data - scaling[0]) /scaling[1] 
    elif '-1' in scaling[2] and '0' not in scaling[2]:
        data = (data - scaling[0]) /scaling[1] *2 -1 
    else:
        print('Invalid scaletype given in data_processing.scale_with_shifts, returning raw data') 
    return data


def unscale_data( data, scaling ):
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
    if scaling[2] is None:
        pass
    elif scaling[2] in [ 'single_std1', 'combined_std1']:
        data = data / scaling[1] + scaling[0] 
    elif  'cov' in scaling[2].lower():
        data = (data @ scaling[1]) + scaling[0] 
    elif '0' in scaling[2] and '1' in scaling[2]:
        data = data * scaling[1] + scaling[0] 
    elif '-1' in scaling[2] and '1' in scaling[2]:
        data = (data + 1) * scaling[1]/2 + scaling[0]  
    else:
        print('Invalid scaletype given in data_processing.unscale_data, returning raw data') 
    return data


class CrossValidation():
    """
    Implementation of the k-fold cross validation
    Randomly shuffle the data and put the data into k-batches. If the 
    data has been 'folded' k times, then it is reshuffled
    CAREFUL: a copy of the full data is created inside this object, might
    lead to memory issues!
    -----------
    How to use:
    data_handler = CrossValidation( x, y, k)
    del x, y
    for i in range( n_epochs):
        x_valid, y_valid = next( data_handler)
        for x_train, y_train in data_handler:
            #model training
    """
    def __init__( self, x, y, k=5 ):
        """ 
        Allocate variables and split the data into k-batches/folds
        Parameters:
        -----------
        x:      torch.tensor like
                input data aranged row wise (each row 1 sample)
        y:      torch.tensor like
                output data aranged row wise, must have the same number
                of samples than x
        k:      int, default 5
                into how many folds to split the data, the fold size 
                computes to n_samples/k
        """
        assert k > 1, 'error in CrossValidation, "k" must be larger than 1'
        self.n_samples = y.shape[0]
        self.batchsize = self.n_samples//k
        self.k = k
        self.x = x
        self.y = y
        self.shuffle_set() 
    
    def shuffle_set( self):
        """ Randomly shuffle the data and put it into K sets """
        permutation = np.random.permutation( self.n_samples )
        self.x = self.x[ permutation]
        self.y = self.y[ permutation]
        self.batches = []
        for i in range( self.k-1):
            self.batches.append( (   self.x[i*self.batchsize:(i+1)*self.batchsize],
                                     self.y[i*self.batchsize:(i+1)*self.batchsize] ))
        self.batches.append( (  self.x[(i+1)*self.batchsize: ], 
                                self.y[(i+1)*self.batchsize: ] ))
        self.fold_counter = 0

    def __next__( self):
        """
        Get the next fold and return the validation set
        If K folds have been conducted, the set is reshuffled 
        """
        if self.fold_counter == 0:
            self.current_val = self.batches.pop()
        elif self.fold_counter < self.k:
            self.batches.insert( 0, self.current_val)
            self.current_val = self.batches.pop()
        else: #fold_counter == k or folds
            self.shuffle_set()
            next(self)
        self.fold_counter += 1
        return self.current_val
    
    def __iter__( self):
        """ Iterate over the remaining training set AFTER calling val_set = next(CrossValidation)"""
        return iter( self.batches)





def batch_data( x, y, n_batches, shuffle=True, stochastic=0.0, x_extra=None, y_extra=None, **kwargs):
    """
    Batch all of the given data into <n_batches> and return the data as 
    list of data. The last batch is the largest if the number of
    samples is not integer divisible by 'n_batches' (the last batch is at
    most 'n_batches-1' larger than the other batches)
    Also enables a stochastic chosing of the training samples by ommiting
    different random samples each epoch
    Parameters:
    -----------
    x:              numpy array
                    input data aranged row wise
    y:              numpy array
                    output data/target values aranged row wise
    n_batches:      int
                    number of batches to return
    shuffle:        bool, default True
                    If the data should be shuffled before batching
    stochastic:     float, default 0.5
                    if the data should be stochastically picked, has to be <=1
                    only available if <shuffle> is True
    x_extra:        numpy array or list of arrays, default None
                    additional input data, asserts that len(x) == len( x_extra)
    y_extra:        numpy array, default None
                    additional output data
    **kwargs:       kwargs
                    only here to catch older verions, no functionality given
    Returns:
    -------
    data_batches    list
                    list of (x_batch, y_batch, 'x_extra, y_extra') pairs
                    if x_extra and y_extra are given
    """
    ## input preprocessing and variale allocation
    if x_extra is not None and len( x_extra) == 1:
        x_extra = x_extra[0]
    n_samples = y.shape[0]
    batchsize = int( n_samples // n_batches * (1-stochastic) )
    max_sample = int( n_samples* (1-stochastic) )
    #i = -1 #to catch errors for n_batches == 1
    jj = 0
    batches = []
    ## shuffle all samples if asked for
    if shuffle:
        permutation = np.random.permutation( n_samples )
        x = permute( x, permutation)
        y = permute( y, permutation)
        if isinstance( x_extra, (list, tuple)):
          for i in range( len( x_extra) ):
            x_extra[i] = permute( x_extra[i], permutation)
        elif x_extra is not None:
            x_extra =  permute( x_extra, permutation)
        if y_extra is not None:
            y_extra = permute( y_extra, permutation)
    else:
        stochastic = 0

    ## slice out the batches and put them into lists
    for i in range( n_batches-1):
        current_batch = []
        ii = i*batchsize
        jj = (i+1)*batchsize
        current_batch.extend( ( x[ii:jj], y[ii:jj] ) )
        if isinstance( x_extra, (list, tuple)):
          extra_batches = [] 
          for i in range( len( x_extra) ):
            extra_batches.append( x_extra[i][ii:jj] )
          current_batch.extend( extra_batches )
        elif x_extra is not None:
            current_batch.append( x_extra[ii:jj] )
        if y_extra is not None:
            current_batch.append( y_extra[ii:jj] )
        batches.append( current_batch)
    ## last batch, take the remaining samples 
    current_batch = []
    current_batch.extend( ( x[ jj:max_sample ], y[jj:max_sample ] ))
    if isinstance( x_extra, (list, tuple)):
      extra_batches = [] 
      for i in range( len( x_extra) ):
        extra_batches.append( x_extra[i][jj:max_sample] )
      current_batch.extend( extra_batches )
    elif x_extra is not None:
        current_batch.append( x_extra[ jj:max_sample ] )
    if y_extra is not None: 
        current_batch.append( y_extra[jj:max_sample ] )
    batches.append( current_batch )
    return batches

def permute( x, permutation):
    """
    permute the array x with permutation. Assumes that the data is arranged 
    row wise. This function is used to permute numpy arrays or tensorflow
    tensors
    Parameters:
    -----------
    x:              numpy nd-array like
                    data to be permuted
    permutation:    indexing object
                    any indexing object admissible for np or tf arrays
    Returns:
    --------
    x:              numpy nd-array like
                    basically x[permutation]
    """
    if 'tensorflow' in sys.modules and isinstance( x, (tf.Tensor, tf.Variable) ):
        return tf.gather( x, permutation)
    return x[permutation]


def rotate_images( images, kappa):
    """
    Rotates all the given images to flip the diagonal components of
    kappa and change the sign of the offdiagonal component
    Parameters:
    -----------
    images:     numpy nd-array like
                images of shape (n_samples, n_x, n_y )
    kappa:      numpy nd-array like
                target values of shape (n_samples, 3), given in mandel notation
    Returns:
    --------
    images:     numpy nd-array like
                previous images rotated 90 degree counter clockwise
    kappa:      numpy nd-array like
                corresponding target values 
    """
    kappa = kappa.copy()
    kappa[:,[0,1]] = kappa[:,[1,0]]
    kappa[:,-1] *= -1
    images = np.rot90( images, axes=(1,2) )
    return images, kappa

def augment_periodic_images( images, y, augmentation=0.5, multi_roll=2, x_extra=None, shuffle=False ):
    """
    Augment the periodic image data by rolling some images randomly.
    Augments by n_samples*augmentation and rolls the same images multiple
    times if specified by multi roll, i.e. default arguments take 1/4 of 
    the images for augmentation.  Does not match the number of samples 
    exactly, i.e. n_samples % multi_roll >= 0 then the few augmented 
    samples are just not added.
    Parameters:
    -----------
    images:         numpy nd-array
                    images of shape (n_samples, 'res,olu,tion', n_channels )
    y:              numpy nd-array
                    target value to copy due to augmentation
    augmentation:   float, default 0.5
                    how much % new samples should be added
    multi_roll:     int >= 1, default 2
                    if randomly selected images should be rolled multiple 
                    times and added multiple times to the set
    shuffle:        bool, default False
                    if images should be shuffled. If False then the data 
                    will be returned in order as (images, augmented_images)
    x_extra:        numpy nd-array, default None
                    additional input data which should be copied for augmentation
    Returns:
    --------
    images:         numpy nd-array
                    augmented image data
    y:              numpy nd-array
                    corresponding target values to image data
    """
    #input preprocessing and variable allocation
    if augmentation is None:
      if x_extra is None:
        return images, y
      else:
        return images, y, x_extra
    multi_roll = max( (multi_roll, 1) )
    n_samples = images.shape[0]
    n_augmented = n_samples * augmentation
    permutation = np.random.permutation( n_samples)
    if shuffle:
        images = images[ permutation] 
    n_augmented = int( n_augmented // multi_roll )
    roll_limits = images.shape [1:-1]
    ndim = len( roll_limits)
    low = np.zeros( ndim, dtype=int)
    ## image augmentation by rolling random samples
    images_augm = []
    y_augm = []
    x_augm = []
    # creation of augmented data
    for _ in range( multi_roll): #roll same sample multiple times
        rolling = np.random.randint( low, roll_limits, size=(n_augmented, ndim ) )
        y_augm.append( y[permutation[:n_augmented]] )
        if x_extra is not None:
            x_augm.append( x_extra[permutation[:n_augmented]] )
        for i in range( n_augmented): #roll and append samples
            image = images[ permutation[i]].reshape( 1, *roll_limits, -1)
            images_augm.append( np.roll( image, rolling[i], axis=range( 1, ndim+1) ) )
    #shuffling of augmented data
    permutation = np.random.permutation( n_augmented*multi_roll)
    images_augm = np.concatenate( images_augm, axis=0 )[permutation] 
    y_augm = np.concatenate( y_augm, axis=0)[permutation] 
    if x_extra is not None:
        x_augm = np.concatenate( x_augm, axis=0)[permutation] 
    #optional shuffle of whole set and return
    if shuffle:
        permutation = np.random.permutation( n_samples + n_augmented*multi_roll)
    else:
        permutation = slice(None)
    images = np.concatenate( (images, images_augm), axis=0 )[permutation]
    y = np.concatenate( (y, y_augm), axis=0)[permutation]
    if x_extra is not None:
        x_extra = np.concatenate( [x_extra, x_augm], axis=0)[permutation]
        return images, y, x_extra
    else:
        return images, y


def batch_generator( n_batches, data, shuffle=True):
    """
    Generator/Factory function, yields 'n_batches' batches when called in a for loop
    The last batch is the largest if the number of samples is not integer divisible by 'n_batches'
    (the last batch is at most 'n_batches-1' larger than the other batches)
    things as i want
    Parameters
    ----------
    n_batches       int
                    number of batches to return
    data:           list of tensorflow.tensors
                    Tensorflow tensors which should be batched
    shuffle         bool, default True
                    If the data should be shuffled during batching
    Yields:
    -------
    batches:        tuple of tensorflow tensors
                    all of the tensors batched as in given order 
    """
    n_samples = data[-1].shape[0]
    if shuffle:
        permutation = np.random.permutation( n_samples)
    else:
        permutation = np.arange( n_samples)
    batchsize = int( n_samples// n_batches)
    i         = -1 # set a value that for n_batches=1 it does return the whole set
    for i in range( n_batches-1):
        idx   = permutation[i*batchsize:(i+1)*batchsize]
        batch = []
        for x in data:
            if 'tensorflow' in sys.modules and isinstance( x, (tf.Tensor, tf.Variable) ):
                batch.append( tf.gather( x, idx) )
            else: #its a numpy array
                batch.append( np.take( x, idx, axis=0) )
        yield batch
    idx   = permutation[(i+1)*batchsize:]
    batch = []
    for x in data:
        if 'tensorflow' in sys.modules and isinstance( x, (tf.Tensor, tf.Variable) ):
            batch.append( tf.gather( x, idx) )
        else: #its a numpy array
            batch.append( np.take( x, idx, axis=0) )
    yield batch


def roll_images( data, part=0.5):
    """
    ## Note that this function is executed on the CPU with numpy. In 
    tf_functions there exists an identical function which is gpu compatible##
    Given periodic images of shape (n_samples, n_1, n_2, n_channels)
    randomly roll the <part> in x and y direction.
    Intended use: data augmentation for periodic image data. Use this 
    function while training to have virtually infinite training samples
    (though the feature span of the training samples does not increase
    due to this procedure)
    Parameters:
    -----------
    data:       list of numpy nd-array
                multiple data sets aranged row wise in tensorflow notation
    part:       float, default 0.5
                what proportion of the randomly selected images should
                be rolled
    Returns:
    --------
    None:       each array in data will be changed in place
    """
    data = [data] if not isinstance( data, list) else data
    n_images = data[0].shape[0]
    n_roll   = int( n_images*part )
    img_dim  = data[0].shape[1:3]
    max_roll = min( img_dim)
    indices  = np.random.permutation( np.arange( n_images))[:n_roll]
    roll     = np.random.uniform( 0, max_roll, size=(n_roll, len(img_dim) ))
    j = 0
    for i in indices:
        for x in data:
            x[i] = np.roll( x[i], roll[j], axis=[0,1] ) 
        j += 1





def slice_args( start, stop, *args):
    """
    Slices all args to the same indices given in <start> and <stop>. Only
    slices numpy-ndarrays and tensorflow.tensors, if other datatypes are 
    given they are returned as passed.
    Parameters:
    -----------
    start:  int
            starting index used for slicing
    stop:   int
            end index used for slicing
    *args:  unspecified amount of (preferably) array-likes.
    Returns:
    --------
    *sliced_args:   tuple of sliced args
                    does copy the sliced arrays into memory 
    """
    try:
        admissible_dtypes = (tf.Tensor, np.ndarray, tf.Variable)
    except:
        admissible_dtypes = np.ndarray
    current_args = []
    for arg in args:
        if isinstance(arg, admissible_dtypes ):
            current_args.append( arg[start:stop])
        else:
            current_args.append( arg)
    return current_args 

def slice_kwargs( start, stop, **kwargs):
    """
    Slices all kwargs to the same indices given in <start> and <stop>. Only
    slices numpy-ndarrays and tensorflow.tensors, if other datatypes are 
    given they are returned as given.
    Parameters:
    -----------
    start:      int
                starting index used for slicing
    stop:       int
                end index used for slicing
    **kwargs:   unspecified amount of key - (preferably) array-likes paires
    Returns:
    --------
    **sliced_kargs: dict of sliced kwargs
                    does copy the sliced arrays into memory, keeps
                    the key-value pairs
    """
    try:
        admissible_dtypes = (tf.Tensor, np.ndarray, tf.Variable)
    except:
        admissible_dtypes = np.ndarray
    current_kwargs = dict()
    for key, kwarg in kwargs.items():
        if isinstance(kwarg, admissible_dtypes ):
            current_kwargs[key] = kwarg[start:stop]
        else:
            current_kwargs[key] = kwarg
    return current_kwargs

