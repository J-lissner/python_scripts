import tensorflow as tf
import numpy as np

#@tf.function decorator not written here, because it does not allow to train multiple ANN
def train_step(x, y, model, loss_object, **model_kwargs):
    """
    Compute the gradients of the loss w.r.t. the trainable variables of the model
    Parameters:
    -----------
    x:              tensorflow.tensor like
                    training input data aranged row wise (each row 1 sample)
    x:              tensorflow.tensor like
                    training output data aranged row wise 
    model:          tf.keras.Model object
                    model which predicts the training data
    loss_object:    tf.keras.losses (or self defined)
                    loss object/function to evaluate the loss
    **model_kwargs: keyworded arguments, no defaults
                    additional kwargs for the model call
    """
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True, **model_kwargs )
        loss = loss_object( y, y_pred )
    gradients = tape.gradient( loss, model.trainable_variables)
    return gradients, loss, y_pred



def relative_mse( y, y_pred, axis=None):
    """
    Compute the relative MSE for the predicted values. The mse 
    is analogue to the frobenius norm for default inputs (and real valued y)
    if axis is specified it can also compute it component wise.
    Note that tf_function decorator makes it unable to store in pickle,
    but might help efficiency
    Parameters:
    -----------
    y:      tensorflow tensor
            target value
    y_pred: tensorflow tensor
            predicted_value
    axis:   bool, default None
            which axis to reduce, reduces to scalar value per default 
    Returns:
    --------
    loss:   scalar or tensorflow tensor
            loss of the prediction
    """
    y_norm = tf.reduce_sum( tf.square(y), axis=axis )
    error = tf.reduce_sum( tf.square(y-y_pred), axis=axis)
    loss = error/y_norm
    return loss**0.5


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
    return return_values


def evaluation( x, y, model, loss_object, **model_kwargs):
    """ Simply predict the model and compute a loss, see parameters of above"""
    y_pred = model( x, training=False, **model_kwargs)
    loss = loss_object( y, y_pred)
    return y_pred, loss


@tf.function
def roll_images( images, part=0.5, shuffle=False):
    """
    Given periodic images of shape (n_samples, n_1, n_2, n_channels)
    randomly roll the <part> in x and y direction.
    Changes the image in place, returns None
    Intended use: data augmentation for periodic image data. Use this 
    function while training to have virtually infinite training samples
    (though the feature span of the training samples does not increase
    due to this procedure)
    Parameters:
    -----------
    images:     tensorflow.tensor
                image data of shape (n_samples, n_1, n_2, n_channels)
    part:       float, default 0.5
                what proportion of the randomly selected images should
                be rolled
    shuffle:    bool, default False
                if the data should be shuffled during rolling
    Returns:
    --------
    images:     tensorflow.tensor
                images with randomly selected <part> randomly rolled
    """
    n_images = images.shape[0]
    n_roll = int( n_images*part )
    img_dim = images.shape[1:3]
    max_roll = min( img_dim)
    indices = tf.random.shuffle( tf.range( n_images, dtype=tf.int32) ) 
    roll = tf.random.uniform( shape=(n_roll, len(img_dim) ), minval=0, maxval=max_roll, dtype=tf.int32 )
    rolled_images = []
    for i in range( n_roll):
        rolled_images.append( tf.roll( images[indices[i]], roll[i], axis=[0,1] )) 
    for i in range( n_roll, n_images):
        rolled_images.append( images[indices[i] ] )
    del images
    images = tf.stack( rolled_images)
    del rolled_images
    if shuffle is False:
        images = tf.gather( images, tf.argsort( indices))
    return images

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


def train(epochs, stopping_delay, x_train, y_train, n_batches, model, optimizer, loss_metric, x_valid, y_valid):
    """
    # This was just one idea, but i think i will scratch that
    def train( model, n_epoch, stopping_delay, spike_delay, n_batches, *data, **minimizers):
    bruh idk if i should have a function with that many parameters
    I think if i just put all the optimizer and loss stuff to kwargs then its managable, also *data, **minimizers
    """
    while i <= epochs and decline <= stopping_delay: 
        batched_data = get.batch_data( x_train, y_train, n_batches )
        batch_loss = []
        k = 0
        for x_batch, y_batch in batched_data:
            gradient, _, y_pred = train_step( x_batch, y_batch, ANN, cost_function) 
            optimizer.apply_gradients( zip(gradient, ANN.trainable_variables) )
            k += 1
            batch_loss.append( loss_metric( y_batch, y_pred) )
        loss_t = np.mean( batch_loss)

        # epoch post processing
        _, loss_v = evaluate( x_valid, y_valid, ANN, loss_metric)
        valid_loss.append( loss_v.numpy() )
        train_loss.append( loss_t ) 
        if valid_loss[-1] < valid_loss[best_epoch]: #TODO more sophisticated criteria
            #current idea: also that the validation is not e.g. 10 times larger than the training error
            best_epoch = i
            decline    = 0
            checkpoint_manager.save() 
        print( 'this is my spike counter{} this is my got_worse {}'.format( spike_counter, decline) )
        if valid_loss[-1] < train_loss[-1]: #while underfit, keep training
            spike_counter += 1
            if spike_counter >= spike_delay:
                print( '!!!!!!!!!!!!!!!! i am resetting my decline because i am underfit !!!!!!!!!!!!!!!!!!1')
                decline = 0
        else:
            spike_counter = 0
        decline += 1
        i += 1
        if i % 500 == 0:
            toc('trained for 500 epochs')
            print( 'current validation loss:\t{:.5f} vs best:\t{:.5f}'.format( valid_loss[-1], valid_loss[best_epoch] ) )
            print( 'vs current train loss:  \t{:.5f} vs best:\t{:.5f}'.format( train_loss[-1], train_loss[best_epoch] ) )
            tic('trained for 500 epochs', True)
    return model, losses

