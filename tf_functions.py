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
def roll_images( images, part=0.5, slave_images=None):
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
    images:     tensorflow.Variable
                image data of shape (n_samples, n_1, n_2, n_channels)
    part:       float, default 0.5
                what proportion of the randomly selected images should
                be rolled
    shuffle:    bool, default False
                if the data should be shuffled during rolling
    slave_images: tensorflow.Variable, default None
                has to be a variable, similar to images
    Returns:
    --------
    images:     tensorflow.tensor
                images with randomly selected <part> randomly rolled
    """
    n_images = images.shape[0]
    n_roll = int( n_images*part )
    img_dim = images.shape[1:3]
    max_roll = min( img_dim)
    indices = tf.random.shuffle( tf.range( n_images, dtype=tf.int32) )[:n_roll]
    roll = tf.random.uniform( shape=(n_roll, len(img_dim) ), minval=0, maxval=max_roll, dtype=tf.int32 )
    tf.debugging.Assert(isinstance( images,  tf.Variable), ['roll_images needs tf.Variable(images)'] ) 
    j = 0
    for i in indices:
        images[i].assign( tf.roll( images[i], roll[j], axis=[0,1] ) )
        if slave_images is not None:
            slave_images[i].assign( tf.roll( slave_images[i], roll[j], axis=[0,1] ) )
        j+= 1


## Generators cannot be tf.function s, it significantly slows down training....
def batch_data( x, y, n_batches, shuffle=True):
    """
    Generator/Factory function, yields 'n_batches' batches when called in a for loop
    The last batch is the largest if the number of samples is not integer divisible by 'n_batches'
    (the last batch is at most 'n_batches-1' larger than the other batches)
    Input:
            x               TF tensor or numpy array
                            input data of the ANN
            x               TF tensor or numpy array
                            output data of the ANN
            n_batches       int
                            number of batches to return
        OPTIONAL
            shuffle=True    bool
                            If the data should be shuffled before batching
    Returns/Yields:
            x_batch         TF tensor or numpy array
                            batched input data
            y_batch         TF tensor or numpy array 
                            batched output data
    """
    n_samples = y.shape[0]
    if shuffle:
        permutation = tf.random.shuffle( tf.range( n_samples) )
        x = tf.gather( x, permutation)
        y = tf.gather( y, permutation)
    batchsize = tf.math.floor( n_samples/ n_batches)
    i = -1 # set a value that for n_batches=1 it does return the whole set
    for i in range( n_batches-1):
        yield x[ i*batchsize:(i+1)*batchsize], y[ i*batchsize:(i+1)*batchsize]
    yield x[(i+1)*batchsize:], y[(i+1)*batchsize:]

