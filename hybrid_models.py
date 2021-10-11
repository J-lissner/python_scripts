import tensorflow as tf
import itertools
from tensorflow.math import ceil
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, Flatten, Concatenate
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic
from other_functions import Cycler


class SeparateVol( Model):
  """
  train a model that is linking the volume fraction directly to the
  output layer and taking all other features in a separate Dense model.
  it is expected to be used as 
  model.pretrain_vol( *args) #has to be done in main code because of bug
  model.freeze_vol()
  """
  def __init__( self, n_output, *args, **kwargs):
    super( SeparateVol, self).__init__()
    self.build_vol( n_output)
    self.build_regressor( n_output)


  def build_vol(self, n_output, n_neurons=5, activation='selu'):
    """ 
    build the architecture of the upper bypass layer which connects the
    first input neuron directly to the output layer. it has one hidden
    layer inbetween to allow for some nonlinearity.
    parameters:
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
    self.vol_part.append( Dense( n_output, activation=None ) ) 


  def build_regressor(self, n_output, neurons=[32,32,16,16], batch_normalization=True, activation='selu'): 
    """
    build the architecture of the remaining Dense model.
    parameters:
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
    self.regressor.append( Dense( n_output) )

                

  def freeze_vol( self):
      for layer in self.vol_part:
          layer.trainable=False



  def pretrain_vol( self, x_train, y_train, x_valid=None, y_valid=None, freeze=True, n_epochs=150, **trainers):
    """
    NOTE: before calling this function the model has to be called that
    it is precompiled. Otherwise no training happens.
    Validation data is required to recover the best model. Otherwise it 
    just trains n_epochs and takes the last model.
    pretrain only the 'upper part' which predicts the volume fraction'
    parameters:
    -----------
    x/y_train:  torch.tensor or numpy array
                training_data
    x/y_valid:  torch.tensor or numpy array, default None
                validation data #todo
    freeze:     bool, default True
                whether or not to freeze the weights after training
    **trainers: kwargs for the training parameters #see below
    n_epochs:   int, default 150
    """
    ## inputs and preprocessing
    learning_rate =  trainers.pop( 'learning_rate', 0.05 )
    optimizer         = trainers.pop( 'optimizer', tf.keras.optimizers.Adam( learning_rate=learning_rate) )
    loss =  trainers.pop( 'loss',  tf.keras.losses.MeanSquaredError() )
    trainable_variables = self.trainable_variables 
    checkpoint = tf.train.Checkpoint( model=self, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager( checkpoints, '/tmp/', max_to_keep=1)
    #trainable_variables = [ layer.trainable_variables for layer in self.vol_part] #or trainable_weights 
    #trainable_variables = self.vol_part.trainable_variables
    valid_loss = [1]
    best_epoch = 0
    for layer in self.vol_part:
        layer.trainable=True
    ## training
    for i in range( n_epochs):
        with tf.GradientTape() as tape:
            y_pred = self.predict_vol( x_train, training=True )
            train_loss = loss( y_train, y_pred )
        gradients = tape.gradient( train_loss, trainable_variables)
        optimizer.apply_gradients( zip(gradients, trainable_variables) )
        if not (x_valid is None and y_valid is None):
            valid_loss.append( self.loss( y_valid, self.predict_vol( x_valid)).numpy() ) 
            if valid_loss[-1] < valid_loss[best_epoch]:
                checkpoint_manager.save()

    ## restore the best model and freeze the layers
    if not (x_valid is None and y_valid is None):
        checkpoint_manager.restore( checkpoint_manager.latest_checkpoint)
    if freeze:
        for layer in self.vol_part:
            layer.trainable=False
    return valid_loss
 

  def call(self, x, training=False):
    x_vol = self.predict_vol( tf.reshape( x[:,0], (-1, 1) ) )
    x = x[:,1:]
    for layer in self.regressor:
        x = layer(x)
    return x + x_vol

  def predict_vol( self, x, training=False):
    x = tf.reshape( x[:,0], ( -1, 1) )
    for layer in self.vol_part:
        x = layer( x, training=training)
    return x


  def predict( self, x):
    return self( x, training=False )


  def predict_validation( self, x):
    return self( x, training=False )


class ConvoCombo( Model):
  """
  train a model that is linking the volume fraction directly to the
  output layer and taking all other features in a separate Dense model.
  it is expected to be used as 
  model.pretrain_vol( *args) #has to be done in main code because of bug
  model.freeze_vol()
  """
  def __init__( self, n_output, *args, **kwargs):
    super( ConvoCombo, self).__init__()
    self.build_vol( n_output)
    self.build_extractor()
    self.build_regressor( n_output)


  def build_extractor( self, activation='relu'):
    """
    Build the convolutional layers which extract features
    """
    self.extractors =  []
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
    avg_small.append( BatchNormalization() )
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


  def build_regressor(self, n_output, neurons=[32,32,16,16], activation='selu', batch_normalization=True, **architecture_todo): 
    """
    build the architecture of the remaining Dense model.
    parameters:
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
    self.regressor.append( Dense( n_output) )

                

  def build_vol(self, n_output, n_neurons=5, activation='selu'):
    """ 
    build the architecture of the upper bypass layer which connects the
    first input neuron directly to the output layer. it has one hidden
    layer inbetween to allow for some nonlinearity.
    parameters:
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
    self.vol_part.append( Dense( n_output, activation=None ) ) 



  def freeze_vol( self):
    """ freeze the layers of the volume fraction linking to the output layer"""
    for layer in self.vol_part:
        layer.trainable=False



  def pretrain_vol( self, x_train, y_train, x_valid=None, y_valid=None, freeze=True, n_epochs=150, **trainers):
    """
    NOTE: before calling this function the model has to be called that
    it is precompiled. Otherwise no training happens.
    pretrain only the 'upper part' which predicts the volume fraction'
    parameters:
    -----------
    x/y_train:  torch.tensor or numpy array
                training_data
    x/y_valid:  torch.tensor or numpy array, default None
                validation data #todo
    freeze:     bool, default True
                whether or not to freeze the weights after training
    **trainers: kwargs for the training parameters #see below
    n_epochs:   int, default 150
    """
    ## inputs and preprocessing
    learning_rate =  trainers.pop( 'learning_rate', 0.05 )
    optimizer         = trainers.pop( 'optimizer', tf.keras.optimizers.Adam( learning_rate=learning_rate) )
    loss =  trainers.pop( 'loss',  tf.keras.losses.MeanSquaredError() )
    trainable_variables = self.trainable_variables 
    checkpoint = tf.train.Checkpoint( model=self, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager( checkpoint, '/tmp/', max_to_keep=1)
    #trainable_variables = [ layer.trainable_variables for layer in self.vol_part] #or trainable_weights 
    #trainable_variables = self.vol_part.trainable_variables
    valid_loss = [1]
    best_epoch = 0
    for layer in self.vol_part:
        layer.trainable=True
    ## training
    for i in range( n_epochs):
        with tf.GradientTape() as tape:
            y_pred = self.predict_vol( x_train, training=True )
            train_loss = loss( y_train, y_pred )
        gradients = tape.gradient( train_loss, trainable_variables)
        optimizer.apply_gradients( zip(gradients, trainable_variables) )
        if not (x_valid is None and y_valid is None):
            valid_loss.append( loss( y_valid, self.predict_vol( x_valid)).numpy() ) 
            if valid_loss[-1] < valid_loss[best_epoch]:
                checkpoint_manager.save() 
    ## restore the best model and freeze the layers
    if not (x_valid is None and y_valid is None):
        checkpoint.restore( checkpoint_manager.latest_checkpoint)
    if freeze:
        for layer in self.vol_part:
            layer.trainable=False
    return valid_loss
 

  def call(self, x, vol=None, training=False):
    """
    Predict the model outputs given the image inputs.
    If the volume fraction is precomputed, then it is not computed from
    the images inside the call (efficiency purpose during training)
    """
    if vol is None: 
        vol = tf.reshape( tf.reduce_mean( x, axis=[1,2,3] ), (-1, 1) )
    x_vol = self.predict_vol( vol, training )
        
    x = self.extract_features( x, training)
    for layer in self.regressor:
        x = layer( x, training=training) 
    return x + x_vol



  def predict_vol( self, x, training=False):
    """ take the volume fraction and predict the heat conductivity"""
    for layer in self.vol_part:
        x = layer( x, training=training)
    return x


  def predict( self, x):
    return self( x, training=False )


  def predict_validation( self, x):
    return self( x, training=False )

