import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pickle
from abc import ABC, abstractmethod
from datetime import datetime
from tensorflow.math import ceil
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Concatenate, Add, concatenate

import data_processing as get
import tf_functions as tfun
import learner_functions as learn
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, Conv2DTransposePeriodic
from unet_modules import InceptionEncoder, FeatureConcatenator, SidePredictor, LayerWrapper, Predictor, InceptionUpsampler
from other_functions import Cycler, tic, toc


class MultilevelNet( ABC):
  """
  Parent class for the multilevel unets
  """
  def __init__( self, *args, **kwargs):
      super().__init__(*args, **kwags)
      #self.n_levels = n_levels


  def batched_prediction( self, batchsize, *inputs, **kwargs):
      """
      predict the given data in batches and return the prediction
      takes variable inputs because this method is inherited to more
      complicated models.
      Note that it does only work if the return value of 'call'
      is a tensorflow.tensor, not a list
      Parameters:
      -----------
      batchsize:  int
                  how large the batches should be
      *inputs:    list of tf.tensor like
                  input data to predict
      **kwargs:   other keyworded options for the call,
                  also takes input data
      Returns:
      --------
      prediction: tensorflow.tensor
                  prediction of the model when using self.call()
      """
      n_batches =  int(inputs[0].shape[0]// batchsize)
      if n_batches == 1:
          return self( *inputs, **kwargs)
      prediction = []
      n_samples = inputs[0].shape[0] if inputs else kwargs.items()[0].shape[0]
      jj = 0 #to catch 1 batch
      for i in range( n_batches-1):
          ii = i* n_samples//n_batches
          jj = (i+1)* n_samples//n_batches
          sliced_args = get.slice_args( ii, jj, *inputs)
          sliced_kwargs = get.slice_kwargs( ii, jj, **kwargs) 
          prediction.append( self( *sliced_args, **sliced_kwargs ) )
      sliced_args = get.slice_args( jj, None, *inputs)
      sliced_kwargs = get.slice_kwargs( jj, None, **kwargs) 
      prediction.append( self( *sliced_args, **sliced_kwargs ) )
      if isinstance( prediction[0], (list,tuple) ): #multilevel prediction
         prediction = [ concatenate( x, axis=0) for x in zip( *prediction)]  
      else:
          prediction = concatenate( prediction, axis=0) 
      return prediction

  def freeze_all( self, freeze=True):
      """ 
      freeze or unfreeze the whole model by calling upon every method
      that contains the word 'freeze'
      """
      freeze_methods = [method for method in dir( self) if 'freeze' in method.lower()]
      freeze_methods.pop( freeze_methods.index( 'freeze_all') )
      freeze_methods.pop( freeze_methods.index( 'freeze_upto') )
      for method in freeze_methods:
          method = getattr( self, method)
          method( freeze)

  ### Methods required for predictor finetuning
  @abstractmethod
  def predictor_features( self, images, *args, **kwargs):
    """
    Return the input to the 'predictor' in order to save computational
    time when only training the predictor
    Parameters:
    -----------
    images:       tensorflow.tensor
                  input data to the model
    Returns:
    --------
    channels:     list of tensorflow.tensors,
                  *args to pipe to the model.predict method
    """

  def predict_tip( self, *args, training=False, **layer_kwargs ):
    return self.predictor( *args, training=training, **layer_kwargs)
    
  ### methods required for level pretraining
  @abstractmethod
  def freeze_upto( self, freeze_limit, freeze=True):
    """ 
    Freeze up to the current layer, required for for pretraining
    Parameters:
    -----------
    freeze:       bool, default True
                  whether to freeze or unfreeze
    freeze_limit: int, default None
                  up to which level it freezes
    """ 
  
  @abstractmethod
  def freeze_predictor( self, freeze=True):
    """ freeze all the parameters that deliver the prediction on the final level"""

  def pretrain_level( self, level, train_data, valid_data, **kwargs  ):
    """
    pretrain at current <level> given the data. Will assume that the data is
    of original resolution and downscale accordingly. Note that the data
    tuples getting passed will become unusable after this function due to
    list method callin. May take every lower level into consideration for
    gradient computation, but will only validate with the current level.
    Parameters:
    -----------
    level:              int or iterable of ints
                        which level(s) to pre train. Has to be sorted ascendingl
                        CARE: will lead to bugs if (level > self.n_levels).any()
    train_data:         tuple of tf.tensor likes
                        input - output data tuple 
    valid_data:         tuple of tf.tensor likes
                        input - output data tuple for validation purposes, 
                        required for early stopping
    **kwargs:           keyworded argumnets with default settings
        batchsize:      int, default 25
                        size of batch
        loss_weigths:   list of floats, default range( 1, n)
                        how to weight each level when 'level' is an iterable
        n_epochs:       int, default 250
                        how many epochs to pre train at most
        learning_rate:  int or tf...learning rate, default learner_functions.RemoteLR()
                        custom learning rate, defaults to my linear schedule with
                        default parameters, which is relatively high learning rate 
        stopping_delay  int, default 20
                        after how many epochs of no improvement of the validat loss to break
        plateau_threshold:  float, default 0.93
                            how to set the plateaus for early stopping
    Returns:
    --------
    losses:             list of list of floats or None
                        [train_loss, valid_loss]
                        Valid loss is of max(level) and train loss considers all levels
    """
    # model related stuff and data preprocessng, things i might need to adjust
    n_epochs = kwargs.pop( 'n_epochs', 350 )
    stopping_delay = kwargs.pop( 'stopping_delay', 30 )
    batchsize = kwargs.pop( 'batchsize', 25 )
    loss_weights = kwargs.pop( 'loss_weights', range( 1, self.n_levels+2) )
    n_batches = max( 1, train_data[0].shape[0] // batchsize )
    learning_rate = kwargs.pop( 'learning_rate', learn.RemoteLR() )
    optimizer_kwargs = kwargs.pop( 'optimizer_kwargs', dict(weight_decay=5e-4, beta_1=0.85, beta_2=0.90  ) )
    plateau_threshold = kwargs.pop( 'plateau_threshold', 0.93 )
    ## other twiddle parameters
    debug_counter = 25
    optimizer     = tfa.optimizers.AdamW( learning_rate=learning_rate, **optimizer_kwargs)
    cost_function = tf.keras.losses.MeanSquaredError() #optimize with
    loss_metric   = tf.keras.losses.MeanSquaredError() #validate with 
    ### other required automatic parameters
    level = [level] if isinstance( level, int) else level
    highest_level = max( level)
    poolers = []
    for i in level[::-1]: #get the poolers
        if i == self.n_levels:
            poolers.append( lambda x: x) 
        elif i == highest_level:
            poolers.append( AvgPool2DPeriodic( 2**(self.n_levels-i) ) )
        elif i < self.n_levels:
            poolers.insert(0, AvgPool2DPeriodic( 2**(highest_level-i) ) ) 
    y_train = poolers[-1]( train_data.pop(-1) ) #can pool on the highest level anyways
    y_valid = poolers[-1]( valid_data.pop(-1) )  #only validate the highest level
    x_valid = valid_data[0]
    poolers[-1] = lambda x: x #we have downsampled to lowest resolution, now retain function
    ## other tensorflow objects related things
    if isinstance( learning_rate, learn.slashable_lr() ):
        learning_rate.reference_optimizer( optimizer)
        learning_rate.reference_model( self)
    checkpoint          = tf.train.Checkpoint( model=self, optimizer=optimizer)
    ckpt_folder         = '/tmp/ckpt_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    checkpoint_manager  = tf.train.CheckpointManager( checkpoint, ckpt_folder, max_to_keep=1)
    checkpoint_manager.save() 
    ## freeze everything except for the things trainign currently, freezing all for simplicity
    self.freeze_all() 
    self.freeze_upto( freeze_limit=highest_level, freeze=False)
    print( f'    Pretraining: optimizing {len( self.trainable_variables)} layers\
 with {sum([np.prod( x.shape) for x in self.trainable_variables])} parameters in level: {level}' )

    ## loop variables allocation
    overfit = 0
    best_epoch = 0
    losses = [ [], [] ]
    pred_valid = self.batched_prediction( batchsize, x_valid, level=highest_level )
    plateau_loss = plateau_threshold * loss_metric( y_valid, pred_valid )
    ## training
    tic( f'trained level {level}', silent=True )
    tic( f'  ## {debug_counter} epochs', silent=True )
    for i in range( n_epochs):
      epoch_loss = []
      for x_batch, y_batch in tfun.batch_data( n_batches, [train_data[0], y_train ] ):
          with tf.GradientTape() as tape:
              y_pred = self( x_batch, level, training=True )
              if len( level) == 1:
                  batch_loss = cost_function( y_batch, y_pred )
              else:
                  batch_loss = 0
                  y_batch = [layer(y_batch) for layer in poolers]
                  for j in level:
                      batch_loss += loss_weights[j] * cost_function( y_pred[j], y_batch[j] )
          gradient = tape.gradient( batch_loss, self.trainable_variables)
          optimizer.apply_gradients( zip( gradient, self.trainable_variables) )
          epoch_loss.append( batch_loss)
      ## epoch post processing
      pred_valid = self.batched_prediction( batchsize, x_valid, level=max(level) ) #of current level
      losses[0].append( np.mean( epoch_loss))
      losses[1].append( loss_metric( y_valid, pred_valid ).numpy().mean())
      if ((i+1) % debug_counter) == 0:
          toc( f'  ## {debug_counter} epochs', auxiliary=f', total epochs: {i+1} ##')
          print( f'    train loss: {losses[0][-1]:1.4e}, vs best    {losses[0][best_epoch]:1.4e}'  )
          print( f'    valid loss: {losses[1][-1]:1.4e}, vs plateau {plateau_loss:1.4e}' )
          tic( f'  ## {debug_counter} epochs', silent=True)
      ## learning rate adjustments
      if losses[1][-1] < plateau_loss:
          best_epoch = i
          plateau_loss = plateau_threshold * losses[1][-1]
          overfit = 0
          checkpoint_manager.save()
      elif plateau_loss*20 > losses[1][-1]: 
          overfit += 1 #if the current loss is not worse by an order of magnitude
      if overfit >= stopping_delay:
          if learn.is_slashable( learning_rate ) and not learning_rate.allow_stopping:
            learning_rate.slash()
            overfit = 0
            plateau_loss = plateau_loss/plateau_threshold
            plateau_threshold = 0.97 if learning_rate.allow_stopping else plateau_threshold
            plateau_loss = plateau_loss*plateau_threshold
            continue
          break
    toc( f'trained level {level}')
    checkpoint.restore( checkpoint_manager.latest_checkpoint)
    if i == (n_epochs - 1):
        print( f'    pretrained level {level} for the full {n_epochs} epochs' )
    else: 
        print( f'    model converged when pretraining {level} after {i} epochs' )
    del poolers #don't want layers lying around
    return losses



class SlimNet( Model, MultilevelNet):
  """ 
  Clean implementation of the unet, cohesive building blocks 
  and slim evaluation scheme (code wise)
  """
  def __init__( self, n_out, n_levels=4, n_channels=12, channel_function=None, direct_upsampling=False, *args, **kwargs):
    """ 
    Parameters:
    -----------
    n_out:              int,
                        number of feature maps to predict
    n_levels:           int, default 4
                        number of times to downsample (input & prediction)
    n_channels:         int, default 6
                        number of channels (constant) in each level 
    channel_function:   lambda function, default None
                        how to increase the channels in each level, defaults
                        to: lambda( level): n_channels * ( 1+ level/3 ) )
    direct_upsampling:  bool, default False
                        if the prediction should be directly taken for the next level
                        prediction (False), or considered as feature for convolution (True)
    """
    super().__init__( *args, **kwargs)
    if channel_function is None:
        channel_function = lambda level, n_channels: int( n_channels * ( 1 + level/3 ) )
    self.direct_upsampling = direct_upsampling
    self.n_out = n_out
    self.n_levels = n_levels
    self.down_path = []
    self.up_path = []
    self.side_predictors = [ ]
    ## have the branch which gives the side prediction, for now
    ## with constant channel amount
    for i in range( n_levels):
        n_current = channel_function( i, n_channels)
        self.down_path.append( InceptionEncoder( n_current, maxpool=(i==0) ) ) 
        self.up_path.append( FeatureConcatenator( n_current) )
        self.side_predictors.append( SidePredictor( n_current, n_out ) )
    self.up_path = self.up_path[::-1]
    self.side_predictors = self.side_predictors[::-1]
    #predict via inception module
    self.coarse_grainers = [ AvgPool2DPeriodic( 2**(i+1)) for i in range(n_levels)][::-1] #the average pooling layers are required here
    self.coarse_grainers.append( lambda x: x)
    self.full_predictor = True
    self.replace_predictor() #default to slim predictor, swaps the variable above


  def replace_predictor( self):
      try: del self.predictor
      except: pass 
      if not self.full_predictor:
          self.predictor = Predictor( self.n_out)
      else:
          self.predictor = LayerWrapper( Conv2D( self.n_out, kernel_size=1) )
      self.full_predictor = not self.full_predictor 
      print( f"replaced the predictor to be the {['slim','full'][self.full_predictor]} predictor" )

  def save_state( self, savepath):
      """ Reqruired for the saver, which predictor/ internal variables it currently has"""
      if savepath[-1] != '/':
          savepath = savepath + '/'
      save_kwargs = dict( full_predictor=self.full_predictor )
      pickle.dump( save_kwargs, open( savepath + 'model_state.pkl', 'wb' ) )

  def recover_state( self, load_path):
      """ get the state of the internal variables and apply it as the model was dumped """
      if load_path[-1] != '/':
          load_path = load_path + '/'
      load_kwargs = pickle.load( open( load_path + 'model_state.pkl', 'rb' ) )
      self.full_predictor = not load_kwargs['full_predictor']
      self.replace_predictor()
      


      


  def call( self, images, level=False, only_features=False, training=False):
    """
    Predict the images from the specified level.
    If <level> is set to false, only the fine resolution prediction is given
    Parameters:
    images:     tensorflow.tensor
                image data to predict
    level:      iterable of ints, default False
                which level to return, 0 is lowest, self.n_levels is original resolution
                If set to True, it returns all levels, (False only last)
    training:   bool, default False
                if the layers are currently trained
    """
    ## translate multilevel prediction here to a list of integers
    level = range(self.n_levels + 1) if level is True else level 
    level = [level] if not (isinstance( level, bool) or hasattr( level, '__iter__' ) ) else level
    predictions = []
    ## first go down and store everything in a list
    down_path = [ images ] #required for the last level
    for layer in self.down_path:
        down_path.append( layer( down_path[-1], training=training ) )
    ## now go up again (and store) the side prediction
    prediction = None 
    level_features = down_path.pop( -1)
    for i in range( len( self.side_predictors)):
        if self.direct_upsampling: #use upsampled prediction for y = y + y_-1
            prediction, feature_channels = self.side_predictors[i]( self.coarse_grainers[i]( images), level_features, prediction, training=training ) 
        else: #use upsampled predictions as features
            prediction, feature_channels = self.side_predictors[i]( self.coarse_grainers[i]( images), level_features, training=training ) 
        ## conditional check on how to store/return current level
        if level is not False and i in level: 
            predictions.append( prediction)
            if len( level) == 1:  return predictions[0] #same as [-1]
            elif i == max(level): return predictions
        ## keep going up
        if self.direct_upsampling:
            level_features = self.up_path[i]( down_path.pop( -1), feature_channels, training=training) 
        else:
            level_features = self.up_path[i]( down_path.pop( -1), feature_channels, prediction, training=training) 
    # level_features now concatenated all required channels
    if only_features: return level_features  #for finetuning training
    if self.direct_upsampling:
        prediction = self.predictor( level_features, prediction, training=training )
    else:
        prediction = self.predictor( level_features, training=training )
    ## conditional check on how to handle return values
    if level is not False and len(level) > 1:  #may only happen if last level requested
        predictions.append( prediction) 
        return predictions
    return prediction 

  def predictor_features( self, images, *args, **kwargs):
    """
    Return the input to the 'predictor' in order to save computational
    time when only training the predictor
    Parameters:
    -----------
    images:       tensorflow.tensor
                  input data to the model
    Returns:
    --------
    channels:     list of tensorflow.tensors,
                  *args to pipe to the model.predict method
                  Note that the list is required for code template
    """
    return [self( images, only_features=True)]
 


    #self.freeze_upto( freeze_limit=highest_level, freeze=False)
    ## its called like that
  ## freezer functions for my LayerWrapper
  def freeze_upto( self, freeze_limit, freeze=True):
    self.freeze_down_path( freeze)
    self.freeze_up_path( freeze, freeze_limit) 
    self.freeze_side_predictors( freeze, freeze_limit)
    self.freeze_predictor( not (freeze_limit == self.n_levels )  )
  ## freeze functions for each different 
  def freeze_down_path( self, freeze=True):
      for layer in self.down_path:
          layer.freeze( freeze)
  def freeze_up_path( self, freeze=True, up_to=None):
      for layer in self.up_path[:up_to]:
          layer.freeze( freeze)
  def freeze_side_predictors( self, freeze=True, up_to=None):
      if up_to is not None:
          up_to = min( self.n_levels, up_to +1 )
      for layer in self.side_predictors[:up_to]:
          layer.freeze( freeze)
  def freeze_predictor( self, freeze=True):
      self.predictor.freeze( freeze)


class VVEnet( SlimNet):
  """
  Have a unet like structure which has two branches, a branch contributing
  to the side predictions, and another branch which contains high level
  features and does not need any side predictions
  The branch without the side predictions has less channels and can not really
  be trained levelwise
  I will copy the exact same structure from above and take additionally 
  the extra branch with the high level features
  """
  def __init__( self, n_out, n_levels=4, n_channels=12, channel_function=None, *args, **v_kwargs):
    """ 
    CARE: may not pass any kwargs to super
    Parameters:
    -----------
    n_out:              int,
                        number of feature maps to predict
    n_levels:           int, default 4
                        number of times to downsample (input & prediction)
    n_channels:         int, default 6
                        number of channels (constant) in each level 
    channel_function:   lambda function, default None
                        how to increase the channels in each level, defaults
                        to: lambda( level): n_channels * max( 1, (level+1)//2 )
    v_kwargs:           kwargs with default arguments
                        basically the above arguments replaced as 's/[a-z]*_/v_/'
                        If not given, will default to the passed arguments
        v_function:     lambda function, slightly different with 
                        lambda level, n_channels: int( n_channels * ( 1 + (level)/2 ) )
        v_conv:         int, default 3
                        number of convolutional operations on each level
    """
    ## super builds the lower predictive branch
    direct_upsampling = v_kwargs.pop( 'direct_upsampling', False)
    super().__init__( n_out, n_levels, n_channels, channel_function, direct_upsampling, *args )
    ## input processing and default arguments
    v_function = v_kwargs.pop( 'v_function', channel_function)
    v_channels = v_kwargs.pop( 'v_channels', n_channels)
    v_levels = v_kwargs.pop( 'v_levels', n_levels) 
    n_conv = v_kwargs.pop( 'v_conv', 3)  #simply add this many operations to every thingy
    if v_function is None:
        v_function = lambda level, n_channels: int( n_channels * ( 1 + (level)/3 ) )
    ## build the model
    conv_layer = lambda n_channels: Conv2DPeriodic( n_channels, kernel_size=3, activation='selu')
    self.direct_down = LayerWrapper()
    self.direct_up = LayerWrapper()
    self.extra_predictor = LayerWrapper() #n_conv*conv 3x3, then 1x1 
    for i in range( n_conv):
        self.extra_predictor.append( conv_layer( v_channels) )
    self.extra_predictor.append( Conv2D( n_out, kernel_size=1 ) )
    self.bypass = Add()
    for i in range( v_levels):
        self.direct_down.append( InceptionEncoder( v_function(i, v_channels, maxpool=(i==0) ) ) )
        self.direct_up.append(   InceptionUpsampler( v_function(v_levels-i-2, v_channels) ) )
        for j in range( n_conv):
            self.direct_down[-1].append( conv_layer( v_function(i, v_channels) ) )
            self.direct_up[-1].append(   conv_layer( v_function(v_levels-i-2, v_channels) ) )
    self.enable_double() #default behaviour

  ## specific functions for this network layout
  def enable_double( self, enable=True):
    """ enable the switch which makes the high level features contribute """
    self.enabled = enable
    print( f'setting the double V net to be enabled={enable}')
    self.freeze_extrapredictor( not enable)

  def save_state( self, savepath):
      """ Reqruired for the saver, which predictor/ internal variables it currently has"""
      if savepath[-1] != '/':
          savepath = savepath + '/'
      save_kwargs = dict( full_predictor=self.full_predictor, enabled=self.enabled )
      pickle.dump( save_kwargs, open( savepath + 'model_state.pkl', 'wb' ) )

  def recover_state( self, load_path):
      """ get the state of the internal variables and apply it as the model was dumped """
      if load_path[-1] != '/':
          load_path = load_path + '/'
      load_kwargs = pickle.load( open( load_path + 'model_state.pkl', 'rb' ) )
      self.full_predictor = not load_kwargs['full_predictor']
      self.replace_predictor()
      self.enable_doble( load_kwargs['enabled'] )
      

  def high_level_prediction( self, images, training=False):
    down_features = [images]
    for layer in self.direct_down:
        down_features.append( layer( down_features[-1], training=training) )
    features = []
    for layer in self.direct_up:
        features = self.bypass( features + [down_features.pop() ] )
        features = [layer( features, training=training)]
    return self.extra_predictor( features[0])


  def freeze_predictor( self, freeze=True):
      self.predictor.freeze( freeze)
      if self.enabled:
        self.freeze_extrapredictor( freeze)
        
  def freeze_extrapredictor( self, freeze=True):
    self.direct_down.freeze( freeze )
    self.direct_up.freeze( freeze )
    self.extra_predictor.freeze( freeze )


  # other required methods
  def call( self, images, level=False, training=False, *layer_args, **layer_kwargs ):
      prediction = super().call(images, level=level, training=training, *layer_args, **layer_kwargs ) 
      if 'only_features' in layer_kwargs and layer_kwargs['only_features']:
          return prediction
      level = [level] if not (isinstance( level,bool) or hasattr( level, '__iter__' )) else level
      if self.enabled and (level in [False, True] or self.n_levels in level):
        if level is False or not isinstance( prediction, list):
          prediction = prediction + self.high_level_prediction( images, training=training)
        else:
          prediction[-1] = prediction[-1] + self.high_level_prediction( images, training=training)
      return prediction

  def predictor_features(self, images):
      features = super().predictor_features( images )
      features.append( images)
      return features

  def predict_tip( self, features, images, training=False, **layer_kwargs):
    """
    Have the precomputed features of the lower branch and evaluate the whole
    upper branch with the images
    Parameters:
    -----------
    features:     list of tf.tensors
                  contains the features and the original image (in that order)
    images:       tf.tensor
                  the actual image data used for the upper V for prediction
    """
    prediction = super().predict_tip( features, training=training, **layer_kwargs)
    if self.enabled:
        prediction += self.high_level_prediction( images, training=training, **layer_kwargs )
    return prediction








## here i simply define fully convolutional neural networks 
class DoubleUNet(Model, MultilevelNet): 
    ## so i always need to store everything on the downscaling and then i concatenate,
    ## and from the lowest layer upward i can free memory
    ## -> i should do only the 3x3 and store it everywhere
    ## -> and the averagepool + 5x5 only when i am at current layer
    ## 'bugs' which i think persist here:
    ##  - the number of channels does not match on the upsampling operation
  ### model building functions
  def __init__( self, n_out, n_levels=4, n_channels=4, loaded=False, *args, **kwargs):
    """
    Parameters:
    -----------
    n_out:          int
                    number of channels to predict
    n_levels:       int, defaut 4
                    how many times to downsample by factor 2
    n_channels: int, defaut 4
                    how many channels in each level, n_channels scales with resolution,
                    i.e. lowest resolution has 4*4 channels (default arguments) 
    loaded:         bool, default False
                    if the predictor should be replaced with the inception module,
                    i.e. if we invoke a new model or load from a stored one
    """
    kwargs.pop( 'channel_function', 0 ) #pop any other funtions that are nt requried here
    super().__init__( *args, **kwargs )
    self.n_levels = n_levels #how many times downsample
    self.n_out = n_out
    self.down_path = []
    self.processors = []
    self.concatenators = []
    self.upsamplers = []
    self.side_predictors = []
    self.check_layer = lambda x: isinstance( x, tf.keras.layers.Layer )
    self.check_iter = lambda x: hasattr( x, '__iter__')
    ## direct path down
    down_layers = lambda n_channels, kernel_size=3, **kwargs: Conv2DPeriodic( n_channels, kernel_size=kernel_size, strides=2, activation='selu', **kwargs )
    up_layers = lambda n_channels, kernel_size, **kwargs: Conv2DTransposePeriodic( n_channels, kernel_size=kernel_size, strides=2, activation='selu', padding='same', **kwargs)
    conv_1x1 = lambda n_channels, **kwargs: Conv2D( n_channels, kernel_size=1, strides=1, activation='selu', **kwargs )
    ### definition of down and upsampling model
    for i in range( self.n_levels): #from top to bottom
        n_current = (i+1)*n_channels
        down_operations = [down_layers( n_current, name=f'direct_down{i}' ) ]
        down_operations.append(  down_layers( n_current, kernel_size=5 ) )
        if i != 0: #only in lower levels
            down_operations.append( MaxPool2DPeriodic( 2 ) )
        self.down_path.append( [ down_operations]  )
        self.down_path[-1].append( Concatenate() )
        self.down_path[-1].append( conv_1x1( n_current) )
        #self.down_path[-1].append( layer())
    for i in range( self.n_levels, 0, -1): #from bottom to top
        n_current = i*n_channels
        idx = self.n_levels-i #required for names
        ## operations on the coarse grained image on 'before' the right
        self.processors.append( [AvgPool2DPeriodic( 2**(i) )] )  #convolutions on each level to the right
        coarse_grain_processor = []
        coarse_grain_processor.append( conv_1x1( 1)  )
        coarse_grain_processor.append( Conv2DPeriodic( n_current, kernel_size=3, strides=1, activation='selu') )
        coarse_grain_processor.append( Conv2DPeriodic( n_current, kernel_size=5, strides=1, activation='selu') )
        self.processors[-1].append( coarse_grain_processor)
        self.processors[-1].append( Concatenate())
        self.processors[-1].append( conv_1x1( n_current, name=f'img_processor{idx}') )
        ### concatenate down_path, processors and avgpooled image, channel reduction before upsampling
        self.concatenators.append( [Concatenate()] )
        self.concatenators[-1].append( Conv2DPeriodic(  n_current, kernel_size=3, strides=1 ) )
        self.concatenators[-1].append( conv_1x1( n_current, name=f'channel_concatenators{idx}') )
        self.concatenators[-1].append( Concatenate() ) #again image and prior features
        ### Use conv2dtranspose to upsample all feature layers
        upsampler = [up_layers( n_current, 2, name=f'upsampler_{idx}') ]
        upsampler.append( up_layers( n_current, 4) ) #inception like structure
        self.upsamplers.append( [upsampler] )
        self.upsamplers[-1].append( Concatenate() )
        self.upsamplers[-1].append( conv_1x1( n_current ) ) 
        self.upsamplers[-1].append( Concatenate() ) #upsampled and conv2dtransposed channels
        ## side out pass on each level for loss prediction before upsampling
        inception_predictor = [conv_1x1(n_current)]
        inception_predictor.append( Conv2DPeriodic( n_current, kernel_size=3, strides=1, activation='selu' ) )
        inception_predictor.append( Conv2DPeriodic( n_current, kernel_size=5, strides=1, activation='selu' ) )
        inception_predictor.append( MaxPool2DPeriodic( 2, strides=1) )
        self.side_predictors.append( [inception_predictor] )
        self.side_predictors[-1].append( Concatenate() )
        self.side_predictors[-1].append( Conv2D( n_out, kernel_size=1, strides=1, activation=None, name=f'level_{idx}_predictor' ) )
        self.side_predictors[-1].append( UpSampling2D() ) #use upsampling for prediction layers
        ### upsampling layers, parallel passes with 1x1
    ### predictors, concatenation of bypass and convolutions
    self.full_predictor = True
    self.replace_predictor() #default to slim predictor, swaps the variable above
  

  def replace_predictor( self ):
    """
    replace the simple predictor with an inception module
    to be done at the end of training for finetuning
    """
    try: del self.predictor
    except: pass #not yet built
    if not self.full_predictor:
        conv_1x1 = lambda n_channels, **kwargs: Conv2D( n_channels, kernel_size=1, activation='selu')
        self.predictor = LayerWrapper()
        self.predictor.append( Concatenate() )
        #replaced with inception module
        generic_inception = LayerWrapper( [conv_1x1( self.n_out)] )
        generic_inception.append( [conv_1x1( self.n_out), Conv2DPeriodic( self.n_out, kernel_size=3, activation='selu') ] )
        generic_inception.append( [conv_1x1( self.n_out), Conv2DPeriodic( self.n_out, kernel_size=5, activation='selu')  ])
        generic_inception.append( MaxPool2DPeriodic( 2, strides=1 ) )
        self.predictor.append( generic_inception)
        self.predictor.append( Concatenate() )
        self.predictor.append( Conv2D( self.n_out, kernel_size=1, activation=None, name='final_predictor') )
    else:
        self.predictor = LayerWrapper()
        self.predictor.append( Concatenate() )
        self.predictor.append( Conv2D( self.n_out, kernel_size=1, strides=1, activation=None, name='final_predictor') )
    self.full_predictor = not self.full_predictor
    print( f"replaced the predictor to be the {['slim','full'][self.full_predictor]} predictor" )


  def predictor_features( self, images, *args, **kwargs):
    """ return the list of features required for the predict method """
    feature_channels = self.go_down( images) 
    feature_channels = self.go_up( feature_channels) #is a list
    return [images] + feature_channels

  ### Predictors of each block and call
  def go_down( self, images, training=False):
    """
    extract features from the image and transform them through the layers
    while downsampling. Will keep all levels for the bypasses in the up path,
    and orders the levels from lowest resolution to highest resolution.
    """
    levels = [images] #store the image easily accessible, only reference, required for later
    for i in range( self.n_levels):
        levels.insert( 0, self.predict_inception( self.down_path[i], levels[0], training=training )  )
    return levels


  def go_up( self, levels, training=False, multilevel_prediction=[], *args, **kwargs):
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
    multilevel_prediction:  iterable of ints, default [] 
                            which upscaling level(s) of prediction to return. 
                            <self.n_levels> refers to the model prediction. 
                            if the number is smaller than <self.n_levels>, 
                            the final prediction is not given. Required for pretraining.
    Returns:
    --------
    upscaled_channels:  tf.Tensor or list of tf.Tensors
                        predictions on the last or each layer, depending on parameters 
    """
    ## Input preprocessing
    multilevel_prediction = range( self.n_levels) if multilevel_prediction is True else multilevel_prediction
    multilevel_prediction = [multilevel_prediction] if (isinstance( multilevel_prediction, int) 
                            and not isinstance( multilevel_prediction, bool) ) else multilevel_prediction
    multistage_predictions = [] 
    previous_channels = []
    up_to = self.n_levels if not multilevel_prediction else max( multilevel_prediction) +1
    up_to = min( up_to, self.n_levels )
    ## Use the coarse grained image and stored channels from the down path and go upward
    for i in range( up_to):
      coarse_grained = self.processors[i][0](levels[-1] )  #coarse graining of image
      layer_channels = self.predict_inception( self.processors[i][1:], coarse_grained, training=training)
      #concatenate processed image, down_path and previous current up path 
      layer_channels = self.concatenators[i][0]( [layer_channels, levels[i] ] + previous_channels )
      layer_channels = self.predict_inception( self.concatenators[i][1:-1], layer_channels, training=training)
      layer_channels = self.concatenators[i][-1]( [layer_channels, coarse_grained]) 
      ### predictions on the current level, and concatenate it back to the feature list
      level_prediction = self.predict_inception( self.side_predictors[i][:-1], layer_channels, training=training) 
      ### if requested append or return the prediction of current level
      if multilevel_prediction and i in multilevel_prediction:
          if len( multilevel_prediction) == 1:
              return level_prediction
          multistage_predictions.append( level_prediction)
      ## upsampling layers to higher resolution
      level_prediction  = self.side_predictors[i][-1]( level_prediction, training=training ) #upsampled
      layer_channels    = self.predict_inception( self.upsamplers[i][:-1], layer_channels, training=training) #process upsampling
      layer_channels    = self.upsamplers[i][-1]( [layer_channels, level_prediction ] ) #concat
      previous_channels = [layer_channels] #for next loop
    ### Returns the features channels 
    if multilevel_prediction: #list of levels
        multistage_predictions.append( previous_channels)  #previous channels has to be a list
        return multistage_predictions
    else: #only feature channels
        return previous_channels


  def predict_tip( self, images, feature_channels, training=False):
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
    if not isinstance( images, list):
        images = [images]
    if not isinstance( feature_channels,list):
        feature_channels = [feature_channels]
    prediction = self.predictor[0]( feature_channels + images, training=training)
    prediction = self.predict_inception( self.predictor[1:], prediction, training=training)
    return prediction
 

  def predict_inception( self, layers, images, debug=False, *layer_args, **layer_kwargs):
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
            if debug: print( f'my layer is currently a {type(layer)} ' )
            inception_pred = []
            for layer in layer:
                if isinstance( layer, list): #deep inception module
                    if debug: 
                        print( 'i am a deep inception module' ) 
                        print( 'this is my layer', layer[0])
                    inception_pred.append( layer[0](images, *layer_args, **layer_kwargs) )
                    for deeper_layer in layer[1:]:
                        inception_pred[-1] = deeper_layer(inception_pred[-1] )
                else: #only parallel convolutions
                    if debug: print( 'i am a flat inception module' )
                    pred =  layer(images, *layer_args, **layer_kwargs) 
                    inception_pred.append(pred)
            images = inception_pred
        else: #simple layer
            images = layer( images, *layer_args, **layer_kwargs) 
    return images


  def call( self, images, level=False, training=False, *args, **kwargs):
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
    level:      iterable of ints, default False
                which level(s) to return. May not return the final
                prediction if specified by the arguments
    Returns:
    --------
    prediction: tensorflow.tensor or list of tf.tensors
                image data prediction of original resolution with
                self.n_output channels, or a list of predictions when
                multistage losses is specified accordingly.
    """
    if level == [self.n_levels] or level == (self.n_levels,) or level == self.n_levels:
        level = False 
    ## prediction via down and then up path
    predictions = self.go_down( images, training=training)
    predictions = self.go_up( predictions, training=training, multilevel_prediction=level )
    if ( (isinstance( level, int) and not isinstance( level, bool) 
        and self.n_levels != level ) or 
        ( hasattr( level, '__iter__') and len(level) > 0 
            and self.n_levels not in level) ):
        pass #don't evaluate predictor
    elif level:
        predictions[-1] = self.predict_tip( images, predictions[-1], training=training) 
    else: #default arguments, only original resolution prediction
        predictions = self.predict_tip( images, predictions, training=training )
    if isinstance( predictions, list) and len( predictions) == 1:
        return predictions[0]
    return predictions
      

  def freeze_upto( self, freeze_limit, freeze=True):
    """ freeze everything up to the required layer, there is a shit workaround 
    here but i kinda don't want to revisit every freeze function below """
    level = freeze_limit
    freeze_limit = range( min(freeze_limit + 1, self.n_levels) )  
    self.freeze_downscalers( False)
    self.freeze_processors( False, freeze_limit) 
    self.freeze_concatenators( False, freeze_limit) 
    self.freeze_side_predictors( False, freeze_limit)
    self.freeze_upsamplers( False, range( level) ) 

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
    if not self.processors:
        return
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
    if not self.down_path:
        return
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
    if not self.upsamplers:
        return 
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
    if not self.concatenators:
        return 
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
    if not self.side_predictors:
        return
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
   if not self.predictor:
       return
   self.predictor.freeze( freeze) #is a LayerWrapper

