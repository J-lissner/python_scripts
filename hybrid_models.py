import tensorflow as tf
import itertools
from datetime import datetime
from tensorflow.math import ceil
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, Flatten, Concatenate

import data_processing as get
import tf_functions as tfun
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic
from other_functions import Cycler, tic, toc


class VolBypass( Model): #previously called 'SeparateVol'
  """
  train a model that is linking the volume fraction directly to the
  output layer and taking all other features in a separate Dense model.
  it is expected to be used as 
  model.pretrain_vol( *args) #has to be done in main code because of bug
  model.freeze_vol()
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
    super( VolBypass, self).__init__()
    self.n_output = n_output
    self.n_vol = n_vol
    self.vol_slice = slice( 0, n_vol)
    self.feature_slice = slice( n_vol, None)
    self.build_vol( )
    self.build_regressor( )


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
                given its length has to match 'n_neurons'
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

  ##### pretraining and layer freezing #####
  def pretrain_section( self, x_train, y_train, x_valid=None, y_valid=None, n_epochs=None, n_batches=40, predictor=None, roll_x0=False ):
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
    x/y_train:  list of torch.tensor or numpy array
                training_data given in lists such that predictor( *x_train) works
    x/y_valid:  list of torch.tensor or numpy array, default None
                validation data 
    n_epochs:   int, default None
                how many epochs to train the model, trains to convergence per default
    n_batches:  int, default 25
                how many batches to batch the data into
    predictor:  method of self, default self.call
                method to predict specific parts of the model
    roll_x0:    bool, default False
                roll the input data, only makes sense if x_train[i] is images
    Returns:
    --------
    valid_loss: numpy 1d-array
                loss of valid data if given. gives the loss of partial ann
    """
    ## input preprocessing
    if predictor is None:
        predictor = self.call
    if n_epochs is None:
        n_epochs = 20000
        stopping_delay = 50
    else:
        stopping_delay = n_epochs
    if len( x_train) == 1:
        x_train.append( None)
    if roll_x0:
        roll_idx = 0
    else:
        roll_idx = -1
    x_train = list( x_train)

    ## allocation of hardwired default parameters
    roll_interval = 7 
    learning_rate =  0.05 
    optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate) 
    loss =  tf.keras.losses.MeanSquaredError() 
    trainable_variables = self.trainable_variables 
    checkpoint = tf.train.Checkpoint( model=self, optimizer=optimizer)
    ckpt_folder = '/tmp/ckpt_{}'.format(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    checkpoint_manager = tf.train.CheckpointManager( checkpoint, ckpt_folder, max_to_keep=1)
    valid_loss = []
    best_epoch = 0
    overfit = 1
    debug_interval = 15
    ## training until convergence or for n_epochs
    tic( '{}: {} additional epochs'.format( predictor.__name__, debug_interval), silent=True )
    print( '### starting training for {} ###'.format( predictor.__name__ ) )
    for i in range( n_epochs):
      if (x_train[-1] is not None or roll_x0) and ((i+1) % roll_interval == 0 ):
          x_train[roll_idx] = get.roll_images( x_train[roll_idx]  )
      batched_data = get.batch_data( x_train[0], y_train, n_batches, x_extra=x_train[-1] )
      for batch in batched_data:
         y_batch = batch.pop(1)
         with tf.GradientTape() as tape:
            y_pred = predictor( *batch, training=True)
            train_loss = loss( y_batch, y_pred )
         gradients = tape.gradient( train_loss, trainable_variables)
         optimizer.apply_gradients( zip(gradients, trainable_variables) )
      if not (x_valid is None and y_valid is None):
          y_pred = predictor( *x_valid )
          valid_loss.append( loss( y_valid, y_pred).numpy() ) 
          if valid_loss[-1] < valid_loss[best_epoch]:
              checkpoint_manager.save() 
              best_epoch = i
              overfit = 0
          overfit += 1
      if overfit == stopping_delay:
          break
      if (i+1) % debug_interval == 0:
          toc( '{}: {} additional epochs'.format( predictor.__name__, debug_interval) )
          print( 'current partial val loss:  {:.6f}  vs best  {:.6f}'.format( valid_loss[-1], valid_loss[best_epoch] ) )
          tic( '{}: {} additional epochs'.format( predictor.__name__, debug_interval), silent=True )

    ## restore the best model
    if not (x_valid is None and y_valid is None):
        checkpoint.restore( checkpoint_manager.latest_checkpoint)
    return valid_loss


  def freeze_all( self, freeze=True):
      """ freeze or unfreeze the whole model """
      freeze_methods = [method for method in dir( self) if 'freeze' in method.lower()]
      freeze_methods.pop( freeze_methods.index( 'freeze_all') )
      for method in freeze_methods:
          method = getattr( self, method)
          method( freeze)

                
  def freeze_vol( self, freeze=True):
    """ freeze the layers of the volume fraction linking to the output layer"""
    for layer in self.vol_part:
        layer.trainable = not freeze
  
  def freeze_regressor( self, freeze=True): 
    for layer in self.regressor:
        layer.trainable = not freeze


  ##### calls and shadows of calls ######
  def call(self, x, training=False, *args, **kwargs):
    """ call function given a feature vector which has the volume
    fraction in the first dimension, and the remaining features in
    the rest"""
    x_vol = self.predict_vol( tf.reshape( 
        x[:,self.vol_slice], (-1, self.vol_slice.stop) ) )
    x = x[:,self.feature_slice]
    for layer in self.regressor:
        x = layer(x)
    return x + x_vol


  def predict_vol( self, x, training=False, *args, **kwargs):
    """ take the volume fraction and predict the outputs"""
    x = tf.reshape( x[:,self.vol_slice], (-1, self.vol_slice.stop) ) 
    for layer in self.vol_part:
        x = layer( x, training=training)
    return x


  def predict( self, x, training=False, *args, **kwargs):
    """ simply shadow call """
    return self( x, training )


  def predict_validation( self, x, *args, **kwargs):
    """ simply shadow call with training hardwired """
    return self( x, training=False )


class ConvoCombo( VolBypass):
  """
  train a model that is linking the volume fraction directly to the
  output layer and taking the images to train a ConvNet branch
  CAREFUL: the regressor from above which used the features is now used
  to predict the ConvNet features. The next inheriting model will create
  a new 'feature_predictor' to have a predictor with the inception module
  and the hand crafted features
  """
  def __init__( self, n_output, *args, **kwargs):
    super( ConvoCombo, self).__init__( n_output, *args, **kwargs)
    # above command should also build vol and build regressor
    self.build_extractor()
    #self.build_vol( )
    #self.build_regressor( ) #now used for convnet regressor


  def build_extractor( self, activation='relu'):
    """
    Build the convolutional layers which extract features
    """
    self.inception_1 = [ [], [], [], [], [] ]
    block_layer = self.inception_1[0]
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

  
  def freeze_inception( self, freeze=True):
      """ freeze the entirety of all inception modules """
      for module in [self.inception_1, self.poolers]:
          for block in module:
              for layer in block:
                  layer.trainable = not freeze


  def extract_features( self, images, training=False):
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
    x_pool = concatenate( x_pool)
    return concatenate( [x, x_pool])

  def predict_inception( self, images, training=False):
    x = self.extract_features( images, training)
    for layer in self.regressor:
        x = layer( x, training=training) 
    return x


  def call(self, x, vol=None, training=False):
    """
    Predict the model outputs given the image inputs.
    If the volume fraction is precomputed, then it is not computed from
    the images inside the call (efficiency purpose during training)
    """
    if vol is None: 
        vol = tf.reshape( tf.reduce_mean( x, axis=[1,2,3] ), (-1, 1) )
    x_vol = self.predict_vol( vol, training )
    x = self.predict_inception( x, training) 
    return x + x_vol



class DecoupledFeatures( ConvoCombo):
  """ 
  Build a 3 part model where all the different features are treated
  separately, and no connection between the differently acquired 
  features is present
  """ 
  def __init__( self, n_output, bayesian_output=False, *args, **kwargs):
    """
    Parameters:
    -----------
    bayesian_ouptut:    bool, default False #TO BE IMPLEMENTEd
                        whether to use bayesian layers/probability prediction
    """
    super( DecoupledFeatures, self).__init__( n_output, *args, **kwargs)
    self.build_regressor( neurons=[ 120, 90, 50])#inherited to build the connection from CNN to output
    self.bayesian = bayesian_output
    self.build_feature_regressor()

  def build_feature_regressor( self, neurons=[45,32,25], activation='selu', batch_normalization=True ):
    self.feature_regressor = []
    layers = self.feature_regressor
    if self.bayesian:
        pass #TODO TO BE IMPLEMENTED 
    else: 
        layer = lambda n_neuron: Dense( n_neuron, activation=activation) 
    for i in range( len( neurons)):
        layers.append( layer( neurons[i] ) )
        if batch_normalization:
            layers.append( BatchNormalization() )
    layers.append( Dense( self.n_output) )

  def freeze_feature_predictor( self, freeze=True):
    """ freeze or unfreeze the feature predictor """
    for layer in self.feature_regressor:
        layer.trainable = not freeze


  def predict_features( self, x, training=False):
    """ only predict the part taking all of the features """
    x = x[:,self.feature_slice]
    for layer in self.feature_regressor:
        x = layer( x, training=training)
    return x


  def call( self, features, images, training=False):
    # predict the volume fraction, features
    x_vol = self.predict_vol( features, training=training)
    x_features = self.predict_features( features, training=training)
    # predict cnn
    x_cnn = self.extract_features( images, training=training )
    # prediction of the regressor of the cnn features
    for layer in self.regressor:
        x_cnn = layer( x_cnn, training=training) 
    return x_vol + x_features + x_cnn 



class FullCombination( DecoupledFeatures ):
  """ build a model in 4 parts, small vol bypass at the top, small-ish
  sized dense model from the scalar features, small-ish dense from the 
  convolutional features, as well as a model combining all higher
  level features """
  def __init__( self, n_output, bayesian_output=False, *args, **kwargs):
    """
    Parameters:
    -----------
    bayesian_ouptut:    bool, default False #TO BE IMPLEMENTEd
                        whether to use bayesian layers/probability prediction
    """
    super( FullCombination, self).__init__( n_output, *args, **kwargs)
    self.build_regressor( neurons=[ 120, 90, 50])#inherited to build the connection from CNN to output
    self.bayesian = bayesian_output
    self.build_connector() #the remaining build functions are inherited


  def build_connector( self, neurons=[ 160, 100, 60, 30], activation='selu', batch_normalization=True ):
    """ connection between hand crafted features and features from the 
    convolutional branch """
    self.connector = []
    layers = self.connector
    if self.bayesian:
        pass #TODO TO BE IMPLEMENTED 
    else: 
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


  def call( self, features, images, training=False):
    # predict the volume fraction, features
    x_vol = self.predict_vol( features, training=training)
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
        
