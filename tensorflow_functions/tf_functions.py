import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from math import ceil, floor

import learner_functions as learn

from timers import tic, toc
from learner_functions import relative_mse, train_step #backward compability

### General data related functions 
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


## Generators cannot be tf.function s, it significantly slows down training....
def batch_data( batchsize, data, shuffle=True):
    """
    Generator/Factory function, yields 'n_batches' batches when called in a for loop
    The last batch is the smallest if the number of samples is not integer divisible
    by 'n_batches' (yielding the remaining samples)
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
    data = [data] if not isinstance( data, (list, tuple)) else data
    n_samples = [x.shape[0] for x in data]
    if shuffle:
        permutation = tf.random.shuffle( tf.range( max(n_samples), dtype=tf.int32) )
    else:
        permutation = tf.range( max( n_samples), dtype=tf.int32 ) 
    n_batches = int( tf.math.ceil( max( n_samples)/ batchsize) )
    i         = -1 # set a value that for n_batches=1 it does return the whole set
    for i in range( n_batches-1):
        idx   = permutation[i*batchsize:(i+1)*batchsize]
        batch = []
        for x, n_max in zip( data, n_samples):
            batch.append( tf.gather( x, idx % n_max) )
        yield batch
    idx   = permutation[(i+1)*batchsize:]
    batch = []
    for x, n_max in zip( data, n_samples):
        batch.append( tf.gather( x, idx % n_max) )
    yield batch

## general image related functions
def roll_images( data, part=0.5):
    """
    Given periodic images of shape (n_samples, n_1, n_2, n_channels)
    randomly roll the <part> in x and y direction.
    Changes the tensors in place, by rolling randomly drawn samples.
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
    Returns:
    --------
    None:       all the input tensors are changed in place
    """
    data = [data] if not isinstance( data, (list, tuple)) else data
    shape = data[0].shape
    n_images = shape[0]
    n_roll   = int( n_images*part )
    img_dim  = shape[1:-1]
    ndim = len( shape) - 2 #samples, channels commented out
    max_roll = min( img_dim)
    indices  = tf.range( n_images, dtype=tf.int32) 
    indices = tf.random.shuffle( indices)
    indices = indices[:n_roll]
    roll    = tf.random.uniform( shape=(n_roll, len(img_dim) ), minval=0, maxval=max_roll, dtype=tf.int32 )
    for x in data:
        tf.debugging.Assert(isinstance( x,  tf.Variable), ['roll_images needs tf.Variable(images)'] ) 
    for x in data:
        j = 0
        for i in indices:
            x[i].assign( tf.roll( x[i], roll[j], axis=list(range(ndim) ) ) )
        j += 1


## generally usable functions 
def n_trainable( model):
    """
    simply count the number of trainable parameters in a model
    Parameters:
    -----------
    model:      tensorflow.keras.Model
                (custom) model class of a keras inhereted model
    Returns:
    --------
    n_params:   int
                total number of trainable parameters
    """
    return sum([np.prod( x.shape) for x in model.trainable_variables])
def train_model( model, data, valid_data, savepath=None, best_loss=1, call=None, image_idx=None, *call_args, **call_kwargs):
    """
    fully train the passed model to convergence with all the default
    state of the art training paramaters i have come to up until now
    Does assume ALOT of default parameters which are not directly 
    accessible, might implement this another time with kwargs, but like
    i said, there are alot which are required. Assumes a whole lot of
    assumptions on our model, e.g. loss etc.
    Will prolly add some parameters in the kwargs as i go, for whatever is
    of interest at the current time
    Parameters:
    -----------
    model:      tensorflow.keras.Model or inherited model
                model which should be trained to convergence
    data:       tuple of tf.tensorflow tensors
                [input, *(inputs,...), output] data tuple, 
                May only accept one output tensor, must be in last index
    valid_data: tuple of tf.tensorflow tensors
                valid data analogue to data
    savepath:   str, default /tmp/training_RNGNUMBER
                full path to where the checkpoints of the model should be stored
    best_loss:  float, default 1
                reference loss such that the model does not worsen
                after training (for checkpoints)
    call:       callable function, default None
                which function in the model should be considered in the loss
                defaults to the __call__ of the model
    image_idx:  list of ints, default None
                if there is image data which should be rolled every 50th epoch
                specifies the indices in <data> which should be rolled
    *call_args: unspecified arguments which should be passed into the call
    *call_args: specified keyworded arguments which should be passed into the call
    """
    ## hardwired default parameters
    savepath = f'/tmp/training_{np.random.randint(0,1e6)}' if savepath is None else savepath
    n_epochs           = 5000
    debug_counter      = 20
    roll_interval      = 50
    stopping_increment = -10 
    n_up               = 1
    n_down             = 3
    stopping_delay     = 25 - (n_up + n_down)*stopping_increment #such that the last trains the shortest (where the fewest things are happening)
    batchsize          = 50
    plateau_threshold  = 0.93 #for variable learning rate
    stop_threshold     = 0.98
    ## object related invocations
    call             = getattr( model, call) if call is not None else model
    lr_kwargs        = dict( base_lr=1e-3, n_up=n_up, n_down=n_down )
    optimizer_kwargs = dict( weight_decay=1e-5, beta_1=0.8, beta_2=0.85 )
    lr_object        = learn.RemoteLR
    learning_rate    = lr_object( **lr_kwargs)
    cost_function    = tf.keras.losses.MeanSquaredError() #optimize with
    loss_metric      = tf.keras.losses.MeanSquaredError() #validate with
    optimizer        = tfa.optimizers.AdamW( learning_rate=learning_rate, **optimizer_kwargs)
    ### automatic computation of other variables
    n_batches = ceil(data[0].shape[0] / batchsize)
    if isinstance( learning_rate, learn.slashable_lr() ):
        learning_rate.reference_optimizer( optimizer)
        learning_rate.reference_model( model)
    checkpoints        = tf.train.Checkpoint( model=model, optimizer=optimizer)#s[-1])
    checkpoint_manager = tf.train.CheckpointManager( checkpoints, savepath, max_to_keep=3 )
    checkpoint_manager.save() 
    ### initialization of looping/tracking variables
    lr_development = np.zeros( n_epochs)
    plateau_loss   = best_loss
    overfit        = 0
    slash_lr       = 0
    best_epoch     = 0
    train_loss     = []
    valid_loss     = []
    roll_data      = map( data.__getitem__, image_idx) if image_idx is not None else None
    y_valid = valid_data.pop( -1)
    ### training of the model
    tic( 'training the model' )
    tic( f'    trained another {debug_counter} epochs', silent=True )
    for i in range( n_epochs):
      if (i+1)%roll_interval == 0 and image_idx is not None:
          debug_images = data[image_idx[0]].copy()
          roll_images( roll_data )
          tf.debugging.assert_near( debug_images, data[image_idx[0] ], 'image was rolled :)' )
                    #TODO  DEBUG THIS SHIT IF THE ROLLING WORKS
      if learn.is_slashable( learning_rate):
          lr_development[i] = learning_rate.learnrate 
      else:
          lr_development[i] = learning_rate
      epoch_loss = []
      for data_batch in batch_data( n_batches, data ):
          y_batch = data_batch.pop(-1)
          with tf.GradientTape() as tape:
              y_pred     = call( *data_batch, *call_args, **call_kwargs, training=True )
              batch_loss = cost_function( y_batch, y_pred )
          gradient = tape.gradient( batch_loss, model.trainable_variables)
          optimizer.apply_gradients( zip( gradient, model.trainable_variables) )
          epoch_loss.append( batch_loss)
      ## epoch post processing
      train_loss.append( tf.reduce_mean( batch_loss ) )
      pred_valid = call( *valid_data, *call_args, **call_kwargs )
      valid_loss.append( loss_metric( y_valid, pred_valid )  ) 
      if (i+1) % debug_counter == 0:
          toc( f'    trained another {debug_counter} epochs', auxiliary=f', total: {i+1}' )
          tic( f'    trained another {debug_counter} epochs', silent=True )
          print( f'    train loss: {train_loss[-1]:1.4e},  vs best {train_loss[best_epoch]:1.4e}'  )
          print( f'    valid loss: {valid_loss[-1]:1.4e},  vs best {valid_loss[best_epoch]:1.4e}' )
          if learn.is_slashable( learning_rate):
            print( f'    plateau:    {plateau_loss:1.4e}' )
      ## learning rate adjustment
      if learn.is_slashable( learning_rate) and valid_loss[-1] < plateau_loss:  
        plateau_loss = plateau_threshold*valid_loss[-1] 
        slash_lr     = 0
      elif learn.is_slashable( learning_rate): #if only marginal improvement
        slash_lr += 1
        if slash_lr == stopping_delay and not learning_rate.allow_stopping:
            learning_rate.slash()
            plateau_loss = plateau_threshold*valid_loss[-1]
            if learning_rate.phase != 1: #if not on the up path
                stopping_delay += stopping_increment
            slash_lr = 0
            overfit  = 0
      else: #make sure that the model stops upon overfitting
          plateau_loss = best_loss
      ## potential early stopping
      if valid_loss[-1] < best_loss and best_loss > plateau_loss/50:
          best_loss = stop_threshold * valid_loss[-1]
          checkpoint_manager.save() 
          best_epoch = i #we started with 1 entry in the valid loss
          overfit    = 0
      else: 
          overfit += 1 
      if overfit > stopping_delay and best_loss > plateau_loss/50: #just dont wave the flag too early
          if learn.is_slashable( learning_rate) and not learning_rate.allow_stopping:
              print( 'slashing the learning rate because i was overfitting' )
              learning_rate.slash()
              plateau_loss     = valid_loss[-1]
              overfit          = 0
              stopping_delay += stopping_increment
              continue
          break
    tracked_variables = dict( train_loss=train_loss, valid_loss=valid_loss, learning_rate=lr_development[:i]) #the rest i can extract from there
    checkpoints.restore( checkpoint_manager.latest_checkpoint)
    if i == (n_epochs - 1):
        print( f'    trained for the full {n_epochs} epochs' )
    else: 
        print( f'    model converged after {i} epochs' )
    toc( 'training the model' )
    return model, tracked_variables

