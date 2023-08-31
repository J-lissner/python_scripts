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
from general_functions import Cycler, tic, toc



class MultilevelNet( ABC):
  """
  Parent class for the multilevel unets
  """
  def __init__( self, *args, **kwargs):
      super().__init__(*args, **kwags)
      #self.n_levels = n_levels


  def batched_prediction( self, batchsize, *inputs, predictor=None, **kwargs):
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
      predictor:  callable, default None
                  which method to evaluate of the model, defaults to self.__call__
      **kwargs:   other keyworded options for the call,
                  also takes input data
      Returns:
      --------
      prediction: tensorflow.tensor
                  prediction of the model when using self.call()
      """
      ## input processing and variable allocation
      n_batches =  int(inputs[0].shape[0]// batchsize)
      if n_batches == 1:
          return self( *inputs, **kwargs)
      if predictor is None:
          predictor = self
      prediction = []
      n_samples = inputs[0].shape[0] if inputs else kwargs.items()[0].shape[0]
      jj = 0 #to catch 1 batch
      ## predict each batch
      for i in range( n_batches-1):
          ii = i* n_samples//n_batches
          jj = (i+1)* n_samples//n_batches
          sliced_args = get.slice_args( ii, jj, *inputs)
          sliced_kwargs = get.slice_kwargs( ii, jj, **kwargs) 
          prediction.append( predictor( *sliced_args, **sliced_kwargs ) )
      sliced_args = get.slice_args( jj, None, *inputs)
      sliced_kwargs = get.slice_kwargs( jj, None, **kwargs) 
      prediction.append( predictor( *sliced_args, **sliced_kwargs ) )
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
      for x_batch, y_batch in tfun.batch_data( batchsize, [train_data[0], y_train ] ):
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
            if learning_rate.allow_stopping: #simply have very few iterations afterwards
                plateau_loss = plateau_loss/plateau_threshold
                plateau_threshold = 0.98 
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
  def __init__( self, n_out, n_levels=4, n_channels=12, channel_function=None, *args, **kwargs):
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
    """
    super().__init__( *args, **kwargs)
    if channel_function is None:
        channel_function = lambda level, n_channels: int( n_channels * ( 1 + level/3 ) )
    self.n_out = n_out
    self.n_levels = n_levels
    self.down_path = []
    self.up_path = []
    self.side_predictors = [ ]
    ## have the branch which gives the side prediction, for now
    ## with constant channel amount
    for i in range( n_levels):
        n_current = channel_function( i, n_channels)
        self.down_path.append( InceptionEncoder( n_current, maxpool=(i>0) ) ) 
        self.up_path.append( FeatureConcatenator( n_current) )
        self.side_predictors.append( SidePredictor( n_current, n_out ) )
    self.up_path = self.up_path[::-1]
    self.side_predictors = self.side_predictors[::-1]
    #predict via inception module
    self.coarse_grainers = [ AvgPool2DPeriodic( 2**(i+1)) for i in range(n_levels)][::-1] #the average pooling layers are required here
    self.coarse_grainers.append( lambda x: x)
    self.full_predictor = True
    self.n_in = n_channels + n_out + 1 #required for predictor with new normalization
    self.replace_predictor() #default to slim predictor, swaps the variable above


  def replace_predictor( self):
      try: del self.predictor
      except: pass 
      if not self.full_predictor:
          self.predictor = Predictor( self.n_out, self.n_in)
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
        prediction, feature_channels = self.side_predictors[i]( self.coarse_grainers[i]( images), level_features, training=training ) 
        ## conditional check on how to store/return current level
        if level is not False and i in level: 
            predictions.append( prediction)
            if len( level) == 1:  return predictions[0] #same as [-1]
            elif i == max(level): return predictions
        ## keep going up
        level_features = self.up_path[i]( down_path.pop( -1), feature_channels, prediction, training=training) 
    # level_features now concatenated all required channels, also the image from 'down_path'
    if only_features: return level_features  #for finetuning training
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
    super().__init__( n_out, n_levels, n_channels, channel_function, *args )
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
        self.direct_down.append( InceptionEncoder( v_function(i, v_channels), maxpool=(i>0) )) 
        self.direct_up.append(   InceptionUpsampler( v_function(v_levels-i-2, v_channels) ))
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
      self.enable_double( load_kwargs['enabled'] )
      

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





