import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from tensorflow.math import ceil
#from my_models import Model 
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate, Flatten, Concatenate

import data_processing as get
import tf_functions as tfun
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic
from other_functions import Cycler, tic, toc


## here i simply define fully convolutional neural networks



class DoubleUNet(Model):

    ## so i always need to store everything on the downscaling and then i concatenate,
    ## and from the lowest layer upward i can free memory
    ## -> i should do only the 3x3 and store it everywhere
    ## -> and the averagepool + 5x5 only when i am at current layer
    ## for now i will assume that the average pooling is a free gift and i do not need
    ## to pool by 2x2x2x2, but can do 8 -> 4- > 2 (ah wait, same amount of operations) 
  ### model building functions
  def __init__( self, channels_out, n_levels=4, *args, **kwargs):
    super().__init__( *args, **kwargs )
    self.n_levels = n_levels #how many times downsample
    channel_per_down = 4 #how many added channels per downsampling in direct convolution
    self.down_path = []
    self.processors = []
    self.concatenators = []
    self.upsamplers = []
    self.side_predictors = []
    self.check_layer = lambda x: isinstance( x, tf.keras.layers.Layer )
    self.check_iter = lambda x: hasattr( x, '__iter__')
    ## direct path down
    down_layers = lambda n_channels, kernel_size=3, **kwargs: Conv2DPeriodic( n_channels, kernel_size=kernel_size, strides=2, activation='selu', **kwargs )
    up_layers = lambda n_channels, kernel_size, **kwargs: Conv2DTranspose( n_channels, kernel_size=kernel_size, strides=2, activation='selu', padding='same', **kwargs)
    conv_1x1 = lambda n_channels, **kwargs: Conv2D( n_channels, kernel_size=1, strides=1, activation='selu', **kwargs )
    ### definition of down and upsampling model
    for i in range( self.n_levels): #from top to bottom
        n_layers = (i+1)*channel_per_down
        self.down_path.append( [[down_layers( n_layers, name=f'direct_down{i}' ), down_layers( n_layers, kernel_size=5 )]]  )
        self.down_path[-1].append( Concatenate() )
        self.down_path[-1].append( conv_1x1( n_layers) )
        #self.down_path[-1].append( layer())
    for i in range( self.n_levels, 0, -1): #from bottom to top
        ## operations on the coarse grained image
        n_layers = i*channel_per_down
        idx = self.n_levels-i #required for names
        self.processors.append( [AvgPool2DPeriodic( 2**(i) )] )  #convolutions on each level to the right
        self.processors[-1].append( Conv2DPeriodic( n_layers, kernel_size=5, strides=1, name=f'img_processor{idx}') )
        ### concatenate and operate prior channels, reduction before upsampling
        self.concatenators.append( [Concatenate()] )
        self.concatenators[-1].append( Conv2DPeriodic(  n_layers, kernel_size=3, strides=1 ) )
        self.concatenators[-1].append( conv_1x1( n_layers, name=f'channel_concatenators{idx}') )
        self.concatenators[-1].append( Concatenate() )
        ## side out pass on each level for loss prediction before upsampling
        self.side_predictors.append( [Conv2DPeriodic( n_layers, kernel_size=5, strides=1, activation='selu' ) ] )
        self.side_predictors[-1].append( [Conv2DPeriodic( n_layers, kernel_size=3, strides=1, activation=None), #inception module
                                Conv2DPeriodic( n_layers, kernel_size=5, strides=1, activation=None)] )
        self.side_predictors[-1].append( Concatenate() )
        self.side_predictors[-1].append( Conv2D( channels_out, kernel_size=1, strides=1, activation=None, name=f'level_{idx}_predictor' ) )
        self.side_predictors[-1].append( Concatenate() )
        ### upsampling layers, parallel passes with 1x1
        #inception like structure
        self.upsamplers.append( [[up_layers( (i)*channel_per_down, 3, name=f'upsampler_{idx}'),up_layers( (i)*channel_per_down, 5)]] )
        self.upsamplers[-1].append( Concatenate() )
        self.upsamplers[-1].append( conv_1x1( n_layers ) ) 
    ### concatenate another bypass and all prior channels, then predict it
    self.build_predictor( channels_out)
  

  def build_predictor( self, n_predict):
    """ 
    build the predictor which gives the final prediction
    n_predict:  int, how many channels to predict
    """
    conv_1x1 = lambda n_channels, **kwargs: Conv2D( n_channels, kernel_size=1, strides=1, activation='selu', **kwargs )
    n_channels = int( max( 1, ceil(2*n_predict/3) ) )
    self.predictor = []
    self.predictor.append( Concatenate() )
    self.predictor.append( [conv_1x1( n_channels) ] )
    self.predictor[-1].append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1)  ) 
    self.predictor[-1].append( Conv2DPeriodic( n_channels, kernel_size=5, strides=1)  ) 
    self.predictor.append( Concatenate() )
    self.predictor.append( Conv2D( n_channels, kernel_size=1, strides=1, activation=None ) )
    # one more smoothing round with inception module, taking a downsamples thing in account
    self.predictor.append( [Conv2DPeriodic( n_channels, kernel_size=5, strides=1) ] )
    self.predictor[-1].append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1)  )
    downsampled_block = [ MaxPool2DPeriodic( 2) ]
    downsampled_block.append( Conv2DPeriodic( n_channels, kernel_size=5, strides=1)  ) 
    downsampled_block.append( Conv2DTranspose( n_channels, kernel_size=5, strides=2, padding='same')  ) 
    self.predictor[-1].append( downsampled_block )
    self.predictor.append( Concatenate() )
    # final prediction
    self.predictor.append( Conv2D( n_predict, kernel_size=1, strides=1, activation=None, name='final_predictor' ) )
    self.predictor.append( Conv2DPeriodic( n_predict, kernel_size=5, strides=1) )



  ### Predictors of each block and call
  def go_down( self, images, training=False):
    """
    extract features from the image and transform them through the layers
    while downsampling. Will keep all levels for the bypasses in the up path,
    and orders the levels from lowest resolution to highest resolution.
    """
    levels = [images] #store the image easily accessible, only reference, required for later
    for i in range( self.n_levels):
        levels.insert( 0, levels[0] ) #take the previous level for operations
        for layer in self.down_path[i]:  #do the operations on each level
          if isinstance( layer,list): #inception module
              level_prediction = []
              for layer in layer:
                  level_prediction.append( layer(levels[0], training=training ) )
              levels[0] = level_prediction #always write to current level
          else:
              levels[0] = layer( levels[0], training=training) 
    return levels

  def go_up( self, levels, training=False, multilevel_prediction=[], single_return=False, *args, **kwargs):
    """
    Upscale the prediction and concatenate each of the layers.
    If multilevel_prediction is specified then a list of predictions is
    returned matching the number of levels, sorted from lowest to highest
    resolution.
    Parameters:
    -----------
    levels:                 list of tf.Tensor like
                            a list of all channels stored during the path down
    training:               bool, default False
                            whether we are currently training, also the switch to
                            predict and return the intermediate layers
    multilevel_prediction:  list of ints, default [] 
                            up to which level to upscale the prediction, if the number
                            is smaller than <self.n_levels>, then the final prediction is
                            not given. Required for pretraining.
    single_return:          bool, default True:
                            whether to return only the requested level. Multilevel_prediction 
                            must be specified via an int in that case.
    Returns:
    --------
    upscaled_channels:  tf.Tensor or list of tf.Tensors
                        predictions on the last or each layer, depending on parameters 
    """
    multilevel_prediction = range( self.n_levels) if multilevel_prediction is True else multilevel_prediction
    multilevel_prediction = [multilevel_prediction] if (isinstance( multilevel_prediction, int) 
                            and not isinstance( multilevel_prediction, bool) ) else multilevel_prediction
    if single_return and len( multilevel_prediction) > 1:
        raise Exception( "InputError in go_up, 'multilevel_prediction' must be an int if 'single_return' is set True" )
    multistage_predictions = [] 
    previous_channels = []
    ## Coarse idea:
    ## put each level into one list, then put arbitrary depth inside these lists
    for i in range( self.n_levels ):
      coarse_grained = self.processors[i][0](levels[-1] )  #original image
      layer_channels = coarse_grained
      for layer in self.processors[i][1:]: #operate on the coarse grained image
          layer_channels = layer( layer_channels, training=training )  
      #concatenate down_path and current up path and do some operations
      layer_channels = self.concatenators[i][0]( [layer_channels, levels[i] ] + previous_channels )
      for layer in self.concatenators[i][1:-1]:  #omit concatenator
          layer_channels = layer( layer_channels, training=training )
      layer_channels = self.concatenators[i][-1]( [layer_channels, coarse_grained]) #add the coarse grained image for upsampling/prediction
      ### predictions on the current level, and concatenate it back to the feature list
      level_prediction = self.side_predictors[i][0]( layer_channels)
      #for layer in self.side_predictors[i][1:-1]: #omit concatenator
      #  level_prediction = layer( level_prediction) 
      level_prediction = self.predict_inception( self.side_predictors[i][1:-1], level_prediction, training=training)
      layer_channels = self.side_predictors[i][-1]( [layer_channels, level_prediction] ) #concatenator
      ### if requested append or return the prediction of current level
      if multilevel_prediction and i in multilevel_prediction:
          if single_return:
              return level_prediction
          multistage_predictions.append( level_prediction)
      ## upsampling layers to higher resolution
      layer_channels = self.predict_inception( self.upsamplers[i], layer_channels, training=training)
      ## prepare for next loop
      previous_channels = [layer_channels]
    if multilevel_prediction:
        multistage_predictions.append( previous_channels)  #previous channels has to be a list
        return multistage_predictions
    else:
        return previous_channels


  def predict( self, images, feature_channels, training=False):
    """
    Return the final prediction of the model by using the images in a 1:1 bypass 
    as well as the high level features derived from the U in the net.
    Parameters:
    -----------
    images:     tf.tensor like of shape 
                image data, must contain 4 channels
    feature_channels:   list of tf.tensor_like
                        features obtained from the U in the model
    training:           bool, default False
                        if we are currently training (@gradients) 
    Returns:
    --------
    prediction: tf.tensor or list of tf.tensor
                final prediction of the model on the original resolution
    """
    #concat image and channels
    prediction = self.predictor[0]( feature_channels + [images])
    prediction = self.predict_inception( self.predictor[1:], prediction, training=training)
    return prediction


  def predict_inception( self, layers, images, *layer_args, **layer_kwargs):
    """
    predict the current images using the layers with the <images> data.
    This function takes in any layers put into list up to 1 inception
    module (no nested inception modules) with arbitrary depth in each
    branch
    Parameters:
    -----------
    layers:     list or nested list of tf.keras.Layers
                layers to conduct the prediction with
    images:     tensorflow.tensor like
                image data of at least 4 dimensions
    *layer_kw/args: 
                arbitrary inputs directy passed to each layer
    """
    for layer in layers:
        if isinstance( layer,list): #inception module
            inception_pred = []
            for layer in layer:
                if isinstance( layer, list): #deep inception module
                    inception_pred.append( layer[0](images, *layer_args, **layer_kwargs) )
                    for deeper_layer in layer[1:]:
                        inception_pred[-1] = deeper_layer(inception_pred[-1] )
                else: #only parallel convolutions
                    pred =  layer(images, *layer_args, **layer_kwargs) 
                    inception_pred.append(pred)
            images = inception_pred
        else: #simple layer
            images = layer( images, *layer_args, **layer_kwargs) 
    return images


  def call( self, images, training=False, multilevel_prediction=[], *args, **kwargs):
    """
    Predict the given images by the double U-net. The first down pass
    needs to be stored in memory, the up pass tries to parallelize as
    many tensor handlings as possible to consider memory limitations.
    After going up the model does give its final prediction with the
    input image.
    Parameters:
    -----------
    images:     tensorflow.tensor like
                image data to predict
    training:   bool, default False
                if the model is trained currently
    multilevel_losses:  list of ints, default []
                which level(s) to return in addition to the model prediction.
    Returns:
    --------
    prediction: tensorflow.tensor or list of tf.tensors
                image data prediction of original resolution with
                self.n_output channels, or a list of predictions when
                multistage losses is set true.
    """
    #multilevel_prediction = [multilevel_prediction] if isinstance( multilevel_prediction, int) else multilevel_prediction
    predictions = self.go_down( images, training=training)
    predictions = self.go_up( predictions, training=training, multilevel_prediction=multilevel_prediction )
    if multilevel_prediction:
        predictions[-1] = self.predict( images, predictions[-1], training=training) 
    else:
        predictions = self.predict( images, predictions, training=training )
    return predictions
      

  ### Pretraining functions
  def pretrain_level( self, level, train_data, n_epochs=250, n_batches=50, valid_data=None, single_training=True ):
    """
    pretrain at current <level> given the data. Will assume that the data is
    of original resolution and downscale accordingly. Note that the data
    tuples getting passed will become unusable after this function due to
    list method callin. May take every lower level into consideration for
    gradient computation, but will only validate with the current level.
    Parameters:
    -----------
    level:              int
                        (up to) which level to pre train
    train_data:         tuple of tf.tensor likes
                        input - output data tuple 
    n_epochs:           int, default 250
                        how many epochs to pre train at most
    n_batches:          int, default 50
                        how many batches for train set
    train_data:         tuple of tf.tensor likes, default None
                        input - output data tuple for validation purposes, enables early stopping
    single_training:    bool, default True
                        if only the current level should be trained or 
                        the levels below as well
    Returns:
    --------
    valid_loss:         list of floats or None
                        if valid data is given the loss at current level is returned
    """
    tic( f'trained level{level}', silent=True )
    # model related stuff and data preprocessng, things i might need to adjust
    optimizer     = tfa.optimizers.AdamW( weight_decay=1e-5 )
    cost_function = tf.keras.losses.MeanSquaredError() #validate with
    loss_metric   = tf.keras.losses.MeanSquaredError() #validate with
    stopping_delay = 20
    debug_counter = 15

    ### other required static variables
    pooling       = AvgPool2DPeriodic( 2**(self.n_levels - level) )
    if valid_data is not None:
        y_valid = pooling( valid_data.pop(-1) ) 
        x_valid = valid_data[0]
    y_train = pooling( train_data.pop(-1) ) #can pool on the highest level anyways
    del pooling
    poolers = [ lambda x: x]
    if single_training is True:
        [poolers.insert(0, AvgPool2DPeriodic( 2**(i+1) ) ) for i in range( level)]

    ## freeze everything except for the things trainign currently, freezing all for simplicity
    self.freeze_all() #for simplicity kept in the loop
    self.freeze_downscalers( False)
    self.freeze_processors( False, range(level+1)) 
    self.freeze_concatenators( False, range(level+1)) 
    self.freeze_side_predictors( False, range( level+1))
    self.freeze_upsamplers( False, range(level) ) 
    ## loop variables allocation
    worsened = 0
    best_epoch = 0
    valid_loss = []
    train_loss = []
    checkpoint          = tf.train.Checkpoint( model=self, optimizer=optimizer)
    ckpt_folder         = '/tmp/ckpt_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    checkpoint_manager  = tf.train.CheckpointManager( checkpoint, ckpt_folder, max_to_keep=1)
    ## training
    tic( f'trained another {debug_counter} epochs', silent=True )
    for i in range( n_epochs):
      batched_data = get.batch_data( train_data[0], y_train, n_batches, x_extra=train_data[1:] )
      epoch_loss = []
      for x_batch, y_batch in batched_data:
          with tf.GradientTape() as tape:
              y_pred = self.go_down( x_batch, training=True)
              y_pred = self.go_up( y_pred, training=True, multilevel_prediction=level, single_return=single_training )
              if single_training:
                  batch_loss = cost_function( y_pred, y_batch )
              else:
                  batch_loss = 0
                  y_batch = [layer(y_batch) for layer in poolers]
                  for j in range( n_levels):
                      batch_loss += cost_function( y_pred[j], y_batch[j] )
          gradient = tape.gradient( batch_loss, self.trainable_variables)
          optimizer.apply_gradients( zip( gradient, self.trainable_variables) )
          epoch_loss.append( batch_loss)
      train_loss.append( tf.reduce_mean( batch_loss ) )
      if valid_data is not None:
          pred_valid = self.go_down( x_valid, training=False) 
          pred_valid = self.go_up( pred_valid, training=True, multilevel_prediction=level, single_return=True )
          valid_loss.append( loss_metric( pred_valid, y_valid)  )
          if valid_loss[-1] < valid_loss[best_epoch]:
              checkpoint_manager.save() 
              best_epoch = i
              worsened = 0
          else: 
              worsened += 1 
          if worsened > stopping_delay:
              print( f'model converged when pretraining {level} after {i} epochs' )
              break
      if (i+1) % debug_counter == 0:
          toc( f'trained another {debug_counter} epochs' )
          tic( f'trained another {debug_counter} epochs', silent=True )
          print( 'current training loss: {:.6f}, vs best {:.6f}'.format( train_loss[i], train_loss[best_epoch] ) )  
          print( 'current valid loss:    {:.6f}, vs best {:.6f}'.format( valid_loss[i], valid_loss[best_epoch] ) )  
    toc( f'trained level{level}')
    checkpoint.restore( checkpoint_manager.latest_checkpoint)
    del poolers #don't want layers lying around
    return valid_loss

  ### Freezing functions
  def freeze_all( self, freeze=True):
      """ 
      freeze or unfreeze the whole model by calling upon every method
      that contains the word 'freeze'
      """
      freeze_methods = [method for method in dir( self) if 'freeze' in method.lower()]
      freeze_methods.pop( freeze_methods.index( 'freeze_all') )
      for method in freeze_methods:
          method = getattr( self, method)
          method( freeze)

  def freeze_processors( self, freeze=True, level=None):
    """ 
    freeze the layers which process the information on each level 'on the
    right side', can specify which level(s) to freeze by passing <level>
    Parameters:
    -----------
    freeze:     bool, default True
                if False the layers will be unfrozen
    level:      iterator of ints or int, default None
                which level to freeze, defaults to all
    """
    level = range( self.n_levels) if level is None else level
    level = [level] if not self.check_iter(level) else level
    for i in level: #freeze at the requested layer
        if self.check_layer(self.processors[i]):
            self.processors[i].trainable = not freeze
        else: 
            for layer in self.processors[i]:
                layer.trainable = not freeze

  def freeze_downscalers( self, freeze=True, level=None):
    """ 
    freeze the layers downscaling 'to the left side', can be
    specified which level(s) to freeze by passing <level>
    Parameters:
    -----------
    freeze:     bool, default True
                if False the layers will be unfrozen
    level:      iterator of ints or int, default None
                which level to freeze, defaults to all
    """
    level = range( self.n_levels) if level is None else level
    level = [level] if not self.check_iter(level) else level
    for i in level: #freeze at the requested layer
        if self.check_layer(self.down_path[i]):
            self.down_path[i].trainable = not freeze
        else: 
            for layer in self.down_path[i]:
                layer.trainable = not freeze
          
  def freeze_upsamplers( self, freeze=True, level=None):
    """ 
    freeze the layers increasing the resolution on the 'right side'
    , can be specified which level(s) to freeze by passing <level>
    Parameters:
    -----------
    freeze:     bool, default True
                if False the layers will be unfrozen
    level:      iterator of ints or int, default None
                which level to freeze, defaults to all
    """
    level = range( self.n_levels) if level is None else level
    level = [level] if not self.check_iter(level) else level
    for i in level: #freeze at the requested layer
        if self.check_layer(self.upsamplers[i]):
            self.upsamplers[i].trainable = not freeze
        else: 
            for layer in self.upsamplers[i]:
                layer.trainable = not freeze

  def freeze_concatenators( self, freeze=True, level=None):
    """ 
    freeze the layers merging all sources together,
    can specify which level(s) to freeze by passing <level>
    Parameters:
    -----------
    freeze:     bool, default True
                if False the layers will be unfrozen
    level:      iterator of ints or int, default None
                which level to freeze, defaults to all
    """
    level = range( self.n_levels) if level is None else level
    level = [level] if not self.check_iter(level) else level
    for i in level: #freeze at the requested layer
        if self.check_layer(self.concatenators[i]):
            self.concatenators[i].trainable = not freeze
        else: 
            for layer in self.concatenators[i]:
                layer.trainable = not freeze


  def freeze_side_predictors( self, freeze=True, level=None):
    """ 
    freeze the layers skip connecting the left to right on each
    level, can specify which level(s) to freeze by passing <level>
    Parameters:
    -----------
    freeze:     bool, default True
                if False the layers will be unfrozen
    level:      iterator of ints or int, default None
                which level to freeze, defaults to all
    """
    level = range( self.n_levels) if level is None else level
    level = [level] if not self.check_iter(level) else level
    for i in level: #freeze at the requested layer
        if self.check_layer(self.side_predictors[i]):
            self.side_predictors[i].trainable = not freeze
        else: 
            for layer in self.side_predictors[i]:
                layer.trainable = not freeze

  
  def freeze_predictor( self, freeze=True):
     """
     freeze the layers for the final prediction of original
     resolution
     """
     for layer in self.predictor:
         if self.check_layer( layer):
           layer.trainable = not freeze
         else:
           for layer in layer:
               layer.trainable = not freeze




