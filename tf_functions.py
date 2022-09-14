import tensorflow as tf
import numpy as np

from learner_functions import relative_mse, train_step #for now to ensure backward compability


def to_float32( *args, arraytype='numpy'):
    """ 
    convert multiple arrays to float 32, mainly used to suppress warnings
    Parameters:
    -----------
    *args:      unspecified amount of tf.tensors or numpy.nd-arrays
                all must share the same package-type
    arraytype:  str, default 'numpy'
                choose between 'numpy' and 'tf/tensorflow'
                what datastructure you are feeding, i.e. which array structure 
    Returns:
    --------
    *args_converted:    all tensors/arrays converted to float32 values.
    """
    return_values = []
    for arg in args:
      if arraytype == 'numpy':
        return_values.append( arg.astype( np.float32) )
      elif arraytype in ['tf', 'tensorflow']:
        return_values.append( tf.cast( arg, tf.float32) )
    if len( return_values) == 1:
        return return_values[0]
    else:
        return return_values


## the tf.function decorator might has to yeet away, if not using inplace
@tf.function
def roll_images( data, part=0.5, shuffle=False):
    """
    Given periodic images of shape (n_samples, n_1, n_2, n_channels)
    randomly roll the <part> in x and y direction.
    Changes the image in place, returns the images if shuffle is on,
    else only inplace
    Intended use: data augmentation for periodic image data. Use this 
    function while training to have virtually infinite training samples
    (though the feature span of the training samples does not increase
    due to this procedure)
    Parameters:
    -----------
    data:       list of tensorflow.Variable s
                multiple tensors of shape (n_samples, n_1, n_2, n_channels)
    part:       float, default 0.5
                what proportion of the randomly selected images should
                be rolled
    shuffle:    bool, default False
                if the data should be shuffled during rolling
    Returns:
    --------
    None:       all the input tensors are changed in place
    """
    n_images = data[0].shape[0]
    n_roll = int( n_images*part )
    img_dim = data[0].shape[1:3]
    max_roll = min( img_dim)
    indices = tf.range( n_images, dtype=tf.int32) 
    if shuffle is not False:
        indices = tf.random.shuffle( indices)
    indices = indices[:n_roll]
    roll = tf.random.uniform( shape=(n_roll, len(img_dim) ), minval=0, maxval=max_roll, dtype=tf.int32 )
    for x in data:
        tf.debugging.Assert(isinstance( x,  tf.Variable), ['roll_images needs tf.Variable(images)'] ) 
    j = 0
    for i in indices:
        for x in data:
            x[i].assign( tf.roll( x[i], roll[j], axis=[0,1] ) )
        j += 1


## Generators cannot be tf.function s, it significantly slows down training....
def batch_data( n_batches, data, shuffle=True):
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
        permutation = tf.random.shuffle( tf.range( n_samples, dtype=tf.int32) )
    else:
        permutation = tf.range( n_samples, dtype=tf.int32 ) 
    batchsize = int( tf.math.floor( n_samples/ n_batches) )
    i = -1 # set a value that for n_batches=1 it does return the whole set
    for i in range( n_batches-1):
        idx = permutation[i*batchsize:(i+1)*batchsize]
        batch = []
        for x in data:
            batch.append( tf.gather( x, idx) )
        yield batch
    idx = permutation[(i+1)*batchsize:]
    batch = []
    for x in data:
        batch.append( tf.gather( x, idx) )
    yield batch

