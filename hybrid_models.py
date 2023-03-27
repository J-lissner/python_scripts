import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from datetime import datetime
from tensorflow.math import ceil
from my_models import Model #from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, Flatten, Concatenate

import data_processing as get
import tf_functions as tfun
import learner_functions as learn
from conv_layers_old import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic
from other_functions import Cycler, tic, toc


class VolBypass( Model): 
  """
  train a model that is linking the volume fraction directly to the
  output layer and taking all other features in a separate Dense model.
  This model does contain a few template functions required for hybrid modeling
  """
  def __init__( self, n_output, n_vol=1, *args, **kwargs):
    """
    build the default model and set internal variables. Default model can
    be overwritten by calling the 'build_*'method
    Parameters:
    -----------
    n_output:       int,
                    number of target values
    n_vol:          int, default 1
                    number of features taken for the vol bypass
    """
    super( VolBypass, self).__init__(n_output, *args, **kwargs)
    self.n_output      = n_output
    self.n_vol         = n_vol
    self.vol_slice     = slice( 0, n_vol)
    self.feature_slice = slice( min(1, n_vol), None) #everything but the volume fraction
    self.build_vol( )
    self.build_feature_regressor( )


  ##### Building of the architecture subblocks ######
  def build_vol(self, n_neurons=5, activation='selu'):
    """ 
    build the architecture of the upper bypass layer which connects the
    first input neuron directly to the output layer. it has one hidden
    layer inbetween to allow for some nonlinearity.
    Parameters:
    -----------
    n_ouptut:   int
                size of the output layer
    n_neurons:  int, default 5
                how big the hidden layer should be 
    activation: string, default 'selu'
                activation function of the hidden layer
    """
    self.vol_part = []
    if isinstance( n_neurons, int ) and n_neurons > 0:
        self.vol_part.append( Dense( n_neurons, activation=activation ) )
    self.vol_part.append( Dense( self.n_output, activation=None ) ) 


  def build_feature_regressor( self, neurons=[45,32,25], activation='selu', batch_normalization=True ):
    """ dense model which predicts the features derived from the Conv Net """
    self.feature_regressor = []
    layers                 = self.feature_regressor
    layer = lambda n_neuron: Dense( n_neuron, activation=activation) 
    for i in range( len( neurons)):
        layers.append( layer( neurons[i] ) )
        if batch_normalization:
            layers.append( BatchNormalization() )
    layers.append( Dense( self.n_output) )


  ##### pretraining and layer freezing #####
  def pretrain_section( self, train_data, valid_data=None,  predictor=None, **kwargs ):
    """
    NOTE: before calling this function the model has to be called that
    it is precompiled. Otherwise no training happens.
    Validation data is required to recover the best model. Otherwise it 
    just trains n_epochs and takes the last model.
    pretrain only the 'upper part' which predicts the volume fraction'
    It trains the part until convergence. Some meaningful default parameters
    are chosen which can't be overwritten in function call.
    All the freezing of other layers has to be done outside this function
    It assumes that x_train[1] is images.
    Parameters:
    -----------
    train_data: list of tf.tensors or numpy arrays
                training data of the form [*inputs, outputs]
    valid_data: list of torch.tensor or numpy array, default None
                validation data of the form [*inputs, outputs]
    predictor:  method of self, default self.call
                method to predict specific parts of the model
    **kwargs with default arguments:
    learning_rate:  str, default 'constant'
                    if my schedule or a constant learning rate should be used 
    batchsize:      int, default 30
                    how large each batch (for trainign and valid should be
    roll_images:    bool, default False
                    roll the input data, only makes sense if x_train[i] is images
    n_epochs:       int, default 20000
                    how many epochs to train the model, trains to convergence per default
    Returns:
    --------
    valid_loss: numpy 1d-array
                loss of valid data if given. gives the loss of partial ann
    """
    ## input preprocessing and default kwargs
    learning_rate  = kwargs.pop( 'learning_rate', 'constant')
    n_epochs       = kwargs.pop( 'n_epochs', 20000)
    roll_images    = kwargs.pop( 'roll_images', False)
    stopping_delay = kwargs.pop( 'stopping_delay', 50) 
    if predictor is None:
        predictor = self.call
    if roll_images:
        roll_idx = [x.ndim for x in train_data].index( 4) 
    # data allocations
    y_valid = valid_data.pop(-1) if valid_data is not None else None

    ## allocation of hardwired default parameters
    roll_interval       = 10 
    if learning_rate == 'constant':
        plateau_threshold   = 1
        learning_rate       = 0.05 
        optimizer           = tf.keras.optimizers.Adam( learning_rate=learning_rate) 
    else:
        plateau_threshold = 0.93
        learning_rate = learn.RemoteLR()
        optimizer = tfa.optimizers.AdamW( learning_rate=learning_rate, weight_decay=1e-4, beta_1=0.8, beta_2=0.85)
        learning_rate.reference_optimizer( optimizer)
        learning_rate.reference_model( self)
    loss                = tf.keras.losses.MeanSquaredError() 
    trainable_variables = self.trainable_variables 
    checkpoint          = tf.train.Checkpoint( model=self, optimizer=optimizer)
    ckpt_folder         = '/tmp/ckpt_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    checkpoint_manager  = tf.train.CheckpointManager( checkpoint, ckpt_folder, max_to_keep=1)
    valid_loss          = []
    best_epoch          = 0
    plateau_loss        = 1e5
    overfit             = 1
    debug_interval      = 15
    ## training until convergence or for n_epochs
    tic( '{}: {} additional epochs'.format( predictor.__name__, debug_interval), silent=True )
    print( '### starting training for {} ###'.format( predictor.__name__ ) )
    for i in range( n_epochs):
      if roll_images and ((i+1) % roll_interval == 0 ):
          tfun.roll_images( train_data[roll_idx]  )
      ## predict the training data data
      for batch in tfun.batch_data( batchsize, train_data ):
         y_batch = batch.pop(-1)
         with tf.GradientTape() as tape:
            y_pred     = predictor( *batch, training=True)
            train_loss = loss( y_batch, y_pred )
         gradients = tape.gradient( train_loss, trainable_variables)
         optimizer.apply_gradients( zip(gradients, trainable_variables) )
      ## predict the validation data
      if y_valid is not None:
        y_pred = self.batched_prediction( batchsize, *valid_data, predictor=predictor )
        valid_loss.append( loss( y_valid, y_pred).numpy() ) 
        ## epoch post processing
        if valid_loss[-1] < plateau_loss:
          plateau_loss = plateau_threshold * valid_loss[-1]
          checkpoint_manager.save() 
          best_epoch = i
          overfit    = 0
        overfit += 1
      if overfit == stopping_delay:
          if learn.is_slashable( learning_rate) and not learning_rate.allow_stopping:
              learning_rate.slash()
              overfit = 0
          else:
              break
      if (i+1) % debug_interval == 0:
          toc( '{}: {} additional epochs'.format( predictor.__name__, debug_interval) )
          print( 'current partial val loss:  {:.6f}  vs best  {:.6f}'.format( valid_loss[-1], valid_loss[best_epoch] ) )
          tic( '{}: {} additional epochs'.format( predictor.__name__, debug_interval), silent=True ) 
    ## restore the best model
    if y_valid is not None:
        checkpoint.restore( checkpoint_manager.latest_checkpoint)
        valid_data.append( y_valid) #put it back into list (inplace operations)
    return valid_loss


  def freeze_feature_predictor( self, freeze=True):
    """ freeze or unfreeze the feature predictor """
    try:
      for layer in self.feature_regressor:
        layer.trainable = not freeze
    except: pass

  def all_but_vol( self, freeze=True):
      """ freeze everything but the volume fraction bypass"""
      self.freeze_all( freeze)
      self.freeze_vol()

  def freeze_vol( self, freeze=True):
    """ freeze the layers of the volume fraction linking to the output layer"""
    for layer in self.vol_part:
        layer.trainable = not freeze
  


  ### predictors and calls
  def predict_vol( self, x, extra_features=[], training=False, *args, **kwargs):
    """ take the volume fraction out of the feature vector and predict the outputs"""
    x = tf.reshape( x[:,self.vol_slice], (-1, self.vol_slice.stop) ) 
    if isinstance( extra_features, (list,tuple)) and len(extra_features) > 0:
        x = concatenate( [x] + extra_features  )
    for layer in self.vol_part:
        x = layer( x, training=training)
    return x


  def predict_features( self, x, training=False):
    """ only predict the part taking all of the features """
    x = x[:,self.feature_slice]
    for layer in self.feature_regressor:
        x = layer( x, training=training)
    return x


  def call(self, x, training=False, *args, **kwargs):
    """ call function given a feature vector which has the volume
    fraction in the first dimension, and the remaining features in
    the rest"""
    vol = tf.reshape( x[:,self.vol_slice], (-1, self.vol_slice.stop) )
    x_vol = self.predict_vol( vol, training=training )
    x_fts = self.predict_features( x, training=training)
    return x_fts + x_vol



class DualInception( VolBypass):
  """
  train a model that is linking the volume fraction directly to the
  output layer and taking the images to train a ConvNet branch
  CAREFUL: the regressor from above which used the features is now used
  to predict the ConvNet features. The next inheriting model will create
  a new 'feature_predictor' to have a predictor with the inception module
  and the hand crafted features
  """
  def __init__( self, n_output, *args, **kwargs):
    super( DualInception, self).__init__( n_output, *args, **kwargs)
    self.build_extractor()
    self.build_regressor() 
    del self.feature_regressor
    #self.build_vol( ) #(inherited)

  def build_regressor(self, neurons=[32,32,16,16], activation='selu', batch_normalization=True, **architecture_todo): 
    """
    build the architecture of the remaining Dense model.
    Parameters:
    -----------
    n_ouptut:   int
                size of the output layer
    n_neurons:  list of ints, default [32,32,16,16]
                how many hidden layers of what size
    activation: string or list of strings, default 'selu'
                activation function of the hidden layer, if a list is 
                given its length has to match <neurons>
    """
    if isinstance( activation, str):
        activation = Cycler( [activation])
    else:
        activation = iter( activation) 
    self.regressor = []
    for n_neurons in neurons:
        self.regressor.append( Dense( n_neurons, activation=next(activation)) )
        if batch_normalization:
            self.regressor.append( BatchNormalization() )
    self.regressor.append( Dense( self.n_output) )


  def build_extractor( self, activation='relu'):
    """
    Build the convolutional layers which extract features
    """
    self.inception_1 = [ [], [], [], [], [] ]
    block_layer      = self.inception_1[0]
    block_layer.append( Conv2DPeriodic( filters=15, kernel_size=17, strides=10, activation=activation)) 
    block_layer.append( MaxPool2DPeriodic( pool_size=2, strides=None) ) #None defaults to pool_size) 
    block_layer.append( BatchNormalization() )
    block_layer.append( Conv2DPeriodic( filters=30, kernel_size=3, strides=2, activation=activation))
    block_layer.append( BatchNormalization() ) 
    #second middle filter generic filter pooling filter
    generic = self.inception_1[1]
    generic.append( Conv2DPeriodic( filters=15, kernel_size=5, strides=3, activation=activation))
    generic.append( MaxPool2DPeriodic( pool_size=2))
    generic.append( BatchNormalization() )
    generic.append( Conv2DPeriodic( filters=30, kernel_size=3, strides=3, activation=activation))
    generic.append( MaxPool2DPeriodic( pool_size=2, padding='valid') )
    generic.append( BatchNormalization() ) 
    # third block with average pooling and a medium sized filter
    avg_medium = self.inception_1[2]
    avg_medium.append( AvgPool2DPeriodic( pool_size=5))
    avg_medium.append( Conv2DPeriodic( filters=20, kernel_size=5, strides=2, activation=activation ) )
    avg_medium.append( BatchNormalization() )
    avg_medium.append( Conv2DPeriodic( filters=40, kernel_size=3, strides=2, activation=activation ) )
    avg_medium.append( MaxPool2DPeriodic( pool_size=2) )
    avg_medium.append( BatchNormalization() ) 
    # third block with average pooling and a medium sized filter
    avg_small = self.inception_1[3]
    avg_small.append( AvgPool2DPeriodic( pool_size=5))
    avg_small.append( Conv2DPeriodic( filters=25, kernel_size=3, strides=2, activation=activation ) )
    avg_small.append( AvgPool2DPeriodic( pool_size=4) )
    avg_small.append( BatchNormalization() )
    #1x1 filter at the end for less features
    convo_1x1 = self.inception_1[4]
    convo_1x1.append( Concatenate())
    convo_1x1.append( Conv2D( filters=50, kernel_size=1, activation=activation ) )
    convo_1x1.append( BatchNormalization())
    convo_1x1.append( Flatten() )
    ###

    self.poolers = [ [], [], [] ]
    # average pooling size 9, fewer larger filters
    big_pool = self.poolers[0]
    big_pool.append( AvgPool2DPeriodic( pool_size=9) )
    big_pool.append( Conv2DPeriodic( filters=20, kernel_size=5, strides=2) )
    big_pool.append( MaxPool2DPeriodic( pool_size=2) )
    big_pool.append( BatchNormalization() )
    big_pool.append( Conv2DPeriodic( filters=30, kernel_size=5, strides=3) )
    big_pool.append( MaxPool2DPeriodic( pool_size=2) )
    big_pool.append( Flatten())
    # pooling size 7, smaller filters and average pooling
    large_pool = self.poolers[1]
    large_pool.append( AvgPool2DPeriodic( pool_size=7) )
    large_pool.append( Conv2DPeriodic( filters=20, kernel_size=3, strides=2) )
    large_pool.append( MaxPool2DPeriodic( pool_size=2) )
    large_pool.append( BatchNormalization() )
    large_pool.append( Conv2DPeriodic( filters=45, kernel_size=5, strides=2) )
    large_pool.append( GlobalAveragePooling2D() )
    small_pool = self.poolers[2]
    small_pool.append( AvgPool2DPeriodic( pool_size=3) )
    small_pool.append( Conv2DPeriodic( filters=10, kernel_size=5, strides=3) )
    small_pool.append( MaxPool2DPeriodic( pool_size=2) )
    small_pool.append( BatchNormalization() )
    small_pool.append( Conv2DPeriodic( filters=15, kernel_size=3, strides=2) )
    small_pool.append( MaxPool2DPeriodic( pool_size=2) )
    small_pool.append( BatchNormalization() )
    small_pool.append( Conv2DPeriodic( filters=20, kernel_size=3, strides=2) )
    small_pool.append( MaxPool2DPeriodic( pool_size=3) )
    small_pool.append( Flatten())

  def freeze_regressor( self, freeze=True): 
    for layer in self.regressor:
        layer.trainable = not freeze
  
  def freeze_inception( self, freeze=True):
      """ freeze the entirety of all inception modules """
      for module in [self.inception_1, self.poolers]:
          for block in module:
              for layer in block:
                  layer.trainable = not freeze


  def extract_features( self, images, extra_features=[], training=False):
    """
    Extract the features based off the images and append extra features if given
    Parameters:
    -----------
    images:         tensor like
                    4 channel input tensor, n_samples, res_x, rex_y, n_channels
    extra_features: list of array-like
                    potential extra features to append to the conv net features
                    can take multiple arrays, have to be of shape n_samples x ?
    training:       bool, default False
                    variable for the layers to set to True during training
    Returns:
    -------- 
    x:              tensorflow.tensor
                    n_samples x n_features feature vector after convolution
    """
    if not isinstance( extra_features, list):
        extra_features = [extra_features]
    x = []
    #first inception layer
    for i in range( len( self.inception_1) -1 ):
        for j in range( len( self.inception_1[i]) ):
            if j == 0:
                x.append( self.inception_1[i][j]( images) )
            else:
                x[i] = self.inception_1[i][j]( x[i] ) 
    #concatenation and 1x1 convo
    for layer in self.inception_1[-1]:
        x = layer( x)
    #second inception layer with pooling yielding scalar valued outputs
    x_pool = []
    for i in range( len( self.poolers)  ):
        for j in range( len( self.poolers[i]) ):
            if j == 0:
                x_pool.append( self.poolers[i][j]( images) )
            else:
                x_pool[i] = self.poolers[i][j]( x_pool[i] ) 
    x_pool = concatenate( x_pool + extra_features)
    return concatenate( [x, x_pool])

  def predict_inception( self, images, extra_features=[], training=False):
    """
    Extract the features based off the images and append extra features if given
    Parameters:
    -----------
    images:         tensor like
                    4 channel input tensor, n_samples, res_x, rex_y, n_channels
    extra_features: list of array-like
                    potential extra features to append to the conv net features
                    can take multiple arrays, have to be of shape n_samples x ?
    training:       bool, default False
                    variable for the layers to set to True during training
    Returns:
    -------- 
    x:              tensorflow.tensor
                    n_samples x n_outputs prediction of the inception part
    """
    x = self.extract_features( images, extra_features, training)
    for layer in self.regressor:
        x = layer( x, training=training) 
    return x


  def call(self, images, vol=None, extra_features=[], training=False):
    """
    Predict the model outputs given the image inputs.
    If the volume fraction is precomputed, then it is not computed from
    the images inside the call (efficiency purpose during training)
    Parameters:
    -----------
    extra_features: list of tensor like
                    additional features to concatenate after convolution layers
    """
    if images.ndim < 4 and vol is not None:
        vol, images = images, vol
    if vol is None: 
        vol = tf.reshape( tf.reduce_mean( images, axis=[1,2,3] ), (-1, 1) )
    x_vol = self.predict_vol( vol, extra_features, training=training )
    x     = self.predict_inception( images, extra_features=extra_features, training=training) 
    return x + x_vol



class DecoupledFeatures( DualInception):
  """ 
  Build a 3 part model where all the different features are treated
  separately, and no connection between the differently acquired 
  features is present
  """ 
  def __init__( self, n_output, *args, **kwargs):
    """ build the three parallel branches with a modified convolution regressor
    besides that it simply takes all things from super()"""
    super( DecoupledFeatures, self).__init__( n_output, *args, **kwargs)
    self.build_feature_regressor()

  def call( self, images, features, extra_features=[], training=False):
    """
    Predict the full model with all subparts
    If variable contrast is trained (i.e. vol slice.stop >1 then the phase contrast
    is automatically passed as extra features for the inception part
    """
    if not extra_features and self.vol_slice.stop == 2:
        extra_features = [ tf.reshape( features[:,1], (-1,1) ) ]
    # predict the volume fraction, features
    x_vol      = self.predict_vol( features, training=training)
    x_features = self.predict_features( features, training=training)
    # predict cnn
    x_cnn = self.extract_features( images, extra_features=extra_features, training=training )
    # prediction of the regressor of the cnn features
    for layer in self.regressor:
        x_cnn = layer( x_cnn, training=training) 
    return x_vol + x_features + x_cnn 



class VariableContrast( DecoupledFeatures):
  """ basically take the model from above and combine the inception and feature
  predictors at the end with the extra features concatenated above """
  def __init__( self, n_output, n_extra=50, *args, **kwargs):
      super( VariableContrast, self).__init__( n_output, *args, **kwargs )
      self.extra_layers = []
      self.extra_layers.append( BatchNormalization() )
      self.extra_layers.append( Dense( n_extra) )
      self.extra_layers.append( BatchNormalization() )
      self.extra_layers.append( Dense( self.n_output) )
  

  def call( self, images, features, extra_features, training=False):
    """
    Predict the full model with all subparts
    requires the extra features which are put on top of both feature 
    layers just before the last layer
    """
    # predict the volume fraction, features
    x_vol      = self.predict_vol( features, training=training)
    x_features = self.predict_features( features, training=training)
    # predict cnn
    x_cnn = self.extract_features( images, training=training )
    # prediction of the regressor of the cnn features
    for layer in self.regressor:
        x_cnn = layer( x_cnn, training=training) 
    x_features = concatenate( [x_cnn, x_features, extra_features])
    for layer in self.extra_layers:
        x_features = layer( x_features, training=training)
    return x_vol + x_features 



class FullCombination( DecoupledFeatures ):
  """ build a model in 4 parts, small vol bypass at the top, small-ish
  sized dense model from the scalar features, small-ish dense from the 
  convolutional features, as well as a model combining all higher
  level features """
  def __init__( self, n_output, *args, **kwargs):
    """ Build the same as from inherited, except for an auxiliary connector
    between features and convolutional features """
    super( FullCombination, self).__init__( n_output, *args, **kwargs)
    self.build_regressor( )#inherited to build the connection from CNN to output
    self.build_connector() #the remaining build functions are inherited


  def build_connector( self, neurons=[ 160, 100, 60, 30], activation='selu', batch_normalization=True ):
    """ connection between hand crafted features and features from the 
    convolutional branch """
    self.connector = []
    layers         = self.connector
    layer = lambda n_neuron: Dense( n_neuron, activation=activation) 
    for i in range( len( neurons)):
        layers.append( layer( neurons[i] ) )
        if batch_normalization:
            layers.append( BatchNormalization() )
    layers.append( Dense( self.n_output) )

  def freeze_connections( self, freeze=True):
    """freeze the full part of the pure CNN part, i.e. feature
    extraction and prediction on the CNN part """
    for layer in self.connector:
        layer.trainable = not freeze


  #def extract_features( inherited)
  #def predict_vol( inherited)
  #def predict_features(inherited)
  
  def predict_intermediate( self, features, images, training=False):
    x_cnn = self.extract_features( images, training=training )
    #prediction of connection between cnn and manual features
    x_combined = concatenate( [features, x_cnn])
    for layer in self.connector:
        x_combined = layer( x_combined, training=training )
    return x_combined


  def call( self, images, features, training=False):
    """
    predict the full model, CARE: EXTRA FEATURES NOT IMPLEMENTED HERE
    (since this model was worse than decoupled features and will not be used subsequently
    """
    # predict the volume fraction, features
    x_vol      = self.predict_vol( features, training=training)
    x_features = self.predict_features( features, training=training)
    # predict cnn
    x_cnn = self.extract_features( images, training=training )
    #prediction of connection between cnn and manual features
    x_combined = concatenate( [features, x_cnn])
    for layer in self.connector:
        x_combined = layer( x_combined, training=training )
    # prediction of the regressor of the cnn features
    for layer in self.regressor:
        x_cnn = layer( x_cnn, training=training) 
    return x_vol + x_features + x_cnn + x_combined
        
