import tensorflow as tf
import numpy as np
import itertools
from tensorflow.math import ceil
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, Flatten, Concatenate, Add
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic
from my_layers import InceptionModule, DeepInception, LayerWrapper, ModularizedDeep
from hybrid_models import VolBypass, FeaturePredictor
from my_models import Model

##################### Convolutional Neural Networks ##################### 
class ModularInceptionII( VolBypass):
    def __init__( self, n_output, pooling=8, n_conv=8, downsampling=64, bypass=False, dense=[50,32,16], *args, padding='periodic', output_activation=None, **kwargs):
        super().__init__( n_output, *args, **kwargs)
        n_modules = 0
        while downsampling > 1:
            n_modules += 1
            downsampling /= pooling
        input_bypass = False
        if bypass:
            input_bypass = Concatenate
            bypass = Add if bypass is True else bypass #if its already a layer dont overwrite
        #first layer should have fewer channels
        n_channels = [ [pooling, min(4*pooling, 64)] ] #linear scaling for first and second layer 
        n_channels.append( [min(4*pooling, 64), 96] ) #depending on pooling not to overflow memory
        n_channels.extend( (n_modules-2)*[96] ) #96 for any deeper layer
        self.conv = [ ModularizedDeep( pooling=pooling, n_conv=n_conv//2, n_channels=n_channels[0], bypass=input_bypass, pool_type='avg', padding=padding )]
        for i in range( 1, n_modules):
            self.conv.append( ModularizedDeep( pooling=pooling, n_conv=n_conv, n_channels=n_channels[i], bypass=bypass, pool_type='max', padding=padding ) )
        self.feature_concatenator= Flatten()

        self.dense = LayerWrapper()
        for n_neuron in dense:
            self.dense.append( Dense( n_neuron, activation='selu') )
            self.dense.append( BatchNormalization() )
        self.dense.append( Dense( n_output, activation=output_activation) )


    def freeze_main( self, freeze=True):
       """
       freeze the entire thing, required for inheritance later
       """
       for layer in self.conv:
           try: layer.freeze( freeze) #its a layer wrapper
           except: pass #its flatten etc.
       self.dense.freeze( freeze)

    def predict_conv( self, images, features=None, x_extra=None, training=False):
        """
        Evaluate the main contribution of this class for the prediction, 
        i.e. the deep inception modules and yield the final prediction
        Parameters:
        -----------
        images:   tensorflow.tensor
                  image data of 4 channels, required for the prediction
        x_extra:      tensorflow.tensor
                      feature vector to append to the dense model
        training: bool, default False
                  flag to inform the model wheter its currently training
        """
        if x_extra is None and self.vol_slice.stop == 2:
            phase_contrast = features[:,1]
            if self.scale_images: #scale input images
                images *= tf.reshape( phase_contrast, (-1, 1,1,1))
            else: #add the  phase contrast in each SnE block
                x_extra = tf.reshape( phase_contrast, (-1,1)) 
        for layer in self.conv:
            images = layer( images, x_extra=x_extra, training=training)
        images = self.feature_concatenator( images)
        return self.dense( images, training=training)

    def call( self, images, x=None, x_extra=None, training=False):
        """ 
        Parameters:
        -----------
        images:       tensorflow.tensor
                      image data of 4 channels required for the prediction
        x:            tensorflow.tensor
                      features required for the volume fraction bypass if enabled 
        x_extra:      tensorflow.tensor
                      feature vector to append to the dense model
        """
        prediction = self.predict_conv( images, x, x_extra, training=training)
        if self.vol_enabled:
            prediction += self.predict_vol( x, training=training) 
        return prediction


class ModularInception( VolBypass): 
  def __init__( self, n_output, n_channels=64, dense=[50,32,16], scale_images=False, module_kwargs=dict(), *args, **kwargs):
      """
      Build a very modal deep inception model which takes 2 (new) modular
      deep inception block in a consecutive manner and has a dense block
      thereafter
      Parameters:
      -----------
      n_output:         int
                        number of neurons to predict
      n_channels:       int, default 32
                        number of output channels per deep inception module
                        scales with $n$ for the $n$-th deep inception block
      dense:            list of ints, default [32,32,16,16]
                        basically number of neurons in the dense part, always
                        uses selu and batch normalization 
      scale_images:    bool, default False
                        switch for the variable phase contrast model if the
                        phase contrast should be added per neuron in SNE, or 
                        if the input image should be scaled by the phase contrast
      module_kwargs:    dict, default dict( n_channel1=[0,n_channels], n_channel2=n_channels)
                        kwargs to pass into both deep inception modules on invocation
      """
      ## input preprocessing
      module_kwargs = module_kwargs.copy()
      n_channel1 = module_kwargs.pop( 'n_channel1', [0, n_channels] )
      n_channel2 = module_kwargs.pop( 'n_channel2', n_channels )
      self.scale_images = scale_images #variable phase contrast implementation

      ##  Definition of model
      super().__init__( n_output, *args, **kwargs)
      self.conv =  [DeepInception( n_channels=n_channel1, **module_kwargs), 
                    DeepInception( n_channels=n_channel2, pooling='max', **module_kwargs)  ]
      self.feature_concatenator = LayerWrapper()
      if len( self.conv) == 1:
        self.feature_concatenator.append( GlobalAveragePooling2D() )
      else:
        self.feature_concatenator.append( Flatten() )
      self.dense = LayerWrapper()
      for n_neuron in dense:
          self.dense.append( Dense( n_neuron, activation='selu') )
          self.dense.append( BatchNormalization() )
      self.dense.append( Dense( n_output, activation=None) ) 
 
  def freeze_main( self, freeze=True):
     """
     freeze the entire thing, required for inheritance later
     """
     for layer in self.conv:
         try: layer.freeze( freeze) #its a layer wrapper
         except: pass #its flatten etc.
     self.feature_concatenator.freeze( freeze)
     self.dense.freeze( freeze)

  def predict_conv( self, images, features=None, x_extra=None, training=False):
      """
      Evaluate the main contribution of this class for the prediction, 
      i.e. the deep inception modules and yield the final prediction
      Parameters:
      -----------
      images:   tensorflow.tensor
                image data of 4 channels, required for the prediction
      x_extra:      tensorflow.tensor
                    feature vector to append to the dense model
      training: bool, default False
                flag to inform the model wheter its currently training
      """
      if x_extra is None and self.vol_slice.stop == 2:
          phase_contrast = features[:,1]
          if self.scale_images: #scale input images
              images *= tf.reshape( phase_contrast, (-1, 1,1,1))
          else: #add the  phase contrast in each SnE block
              x_extra = tf.reshape( phase_contrast, (-1,1)) 
      for layer in self.conv:
          images = layer( images, x_extra=x_extra, training=training)
      images = self.feature_concatenator( images, training=training)
      return self.dense( images, training=training)

  def call( self, images, x=None, x_extra=None, training=False):
      """ 
      Parameters:
      -----------
      images:       tensorflow.tensor
                    image data of 4 channels required for the prediction
      x:            tensorflow.tensor
                    features required for the volume fraction bypass if enabled 
      x_extra:      tensorflow.tensor
                    feature vector to append to the dense model
      """
      prediction = self.predict_conv( images, x, x_extra, training=training)
      if self.vol_enabled:
          prediction += self.predict_vol( x, training=training) 
      return prediction


class InceptionHybrid(FeaturePredictor, ModularInception):
  """ 
  Build the hybrid model using the new inception blocks and the previous
  feature features
  Literally all functionality should be inherited
  """
  def __init__( self, *args, **kwargs):
      super().__init__( *args, **kwargs)

  def call( self, images, x=None, training=False):
    pred = self.predict_features( x, training=training)
    pred += self.predict_conv( images, training=training )
    if self.vol_enabled:
        pred += self.predict_vol( x, training=training )
    return pred


class ArchitectureTest( VolBypass):
    ### here i run super basic architectures and simply test out a few hypothesis
  def __init__( self, n_output, dense=[32,25,16], *args, **kwargs ):
    super().__init__( n_output)
    n_strided = 8
    self.architecture = LayerWrapper()
    self.architecture.append( AvgPool2DPeriodic(2) )
    for i in range( n_strided):
        self.architecture.append( Conv2DPeriodic( 64, kernel_size=3, strides=2) )
        self.architecture.append( Conv2DPeriodic( 64, kernel_size=3) )
        self.architecture.append( Conv2D( 64, kernel_size=1, activation='selu' ) )
    self.architecture.append( Flatten() )
    for n_neuron in dense:
        self.architecture.append( Dense( n_neuron, activation='selu') )
        self.architecture.append( BatchNormalization() )
    self.architecture.append( Dense( n_output, activation=None) )

  def call( self, images, x, training=False, *args, **kwargs):
      return self.architecture( images, training=training) 



class InceptionLike( Model):
  def __init__( self, n_output, input_size, activation=None, n_blocks=1, globalpool=False, *args, **kwargs):
    """
    n_output: int, size output
    input_size: int, size input
    activation: activation function in all layers
    globalpool: if global average pooling should be conducted before flattening 
    """
    super( InceptionLike, self).__init__()
    self.n_blocks = n_blocks
    self.activation = activation
    self.input_size = input_size
    self.n_output = n_output
    self.globalpool = globalpool
    self.build_model()


  def build_model( self):
    #### First inception block
    self.inception_1 = [ [], [], [], [] ]
    block_layer = self.inception_1[0]
    block_layer.append( Conv2DPeriodic( filters=5, kernel_size=17, strides=10, activation=self.activation, input_shape=self.input_size)) 
    block_layer.append( MaxPool2DPeriodic( pool_size=2, strides=None) ) #None defaults to pool_size) 
    block_layer.append( BatchNormalization() )
    block_layer.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=2, activation=self.activation))
    block_layer.append( BatchNormalization() )

    #second middle filter generic filter pooling filter
    generic = self.inception_1[1]
    generic.append( Conv2DPeriodic( filters=5, kernel_size=5, strides=3, activation=self.activation, input_shape=self.input_size))
    generic.append( MaxPool2DPeriodic( pool_size=2))
    generic.append( BatchNormalization() )
    generic.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=3, activation=self.activation))
    generic.append( MaxPool2DPeriodic( pool_size=2, padding='valid') )
    generic.append( BatchNormalization() )

    # third block with average pooling and a medium sized filter
    avg_medium = self.inception_1[2]
    avg_medium.append( AvgPool2DPeriodic( pool_size=5, input_shape = self.input_size))
    avg_medium.append( Conv2DPeriodic( filters=5, kernel_size=5, strides=2, activation=self.activation ) )
    avg_medium.append( BatchNormalization() )
    avg_medium.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=2, activation=self.activation ) )
    avg_medium.append( MaxPool2DPeriodic( pool_size=2) )
    avg_medium.append( BatchNormalization() )

    # third block with average pooling and a medium sized filter
    avg_small = self.inception_1[3]
    avg_small.append( AvgPool2DPeriodic( pool_size=5, input_shape=self.input_size))
    avg_small.append( BatchNormalization() )
    avg_small.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=2, activation=self.activation ) )
    avg_small.append( AvgPool2DPeriodic( pool_size=4) )
    avg_small.append( BatchNormalization() )
    ########

    self.dense = []
    if self.globalpool is True:
        self.dense.append( GlobalAveragePooling2D() )
    self.dense.append( Flatten() )
    self.dense.append( Dense( 128, activation='selu') )
    self.dense.append( BatchNormalization() )
    self.dense.append( Dense( 64, activation='selu') )
    self.dense.append( BatchNormalization() )
    self.dense.append( Dense( 32, activation='selu') )
    self.dense.append( BatchNormalization() ) 
    self.dense.append( Dense( 16, activation='selu') ) 
    self.dense.append( BatchNormalization() ) 
    self.dense.append( Dense( self.n_output) ) 


  def call( self, images):
    #convolutions of feature extractions
    x = []
    for i in range( len( self.inception_1)):
        for j in range( len( self.inception_1[i]) ):
            if j == 0:
                x.append( self.inception_1[i][j]( images) )
            else:
                x[i] = self.inception_1[i][j]( x[i] ) 

    # concatenation after first inception layer
    x = concatenate( x, axis=-1 )#
    if self.n_blocks == 2:
        pass 

    #dense layer prediction
    #x = concatenate( x), vol, k_1, k_2)
    for layer in self.dense:
        x = layer(x)
    return x



class TranslationInvariant( VolBypass):
  """ 
  here we attempt to have a fully translational invariant neural network
  by implementing periodic padding etc. and using globalpool at the end to
  lose the relative position of each convolution feature. 
  It turns out that it does not quite work that way and we only achieve 
  true a priori transltion invariance for no stride and globalpool (which
  is a shit prescription on the model because then the resolution doesn't 
  downsample and we have so many fucking pixels throughout the whole image
  """
  def __init__( self, n_output, n_vol=1, *args, **kwargs):
    super().__init__( n_output, n_vol, *args, **kwargs)
    del self.regressor
    self.build_cnn()
    
  def build_cnn( self):
    """ build two blocks of inception modules with an average globalpooling at the end"""
    stride = 1  #for true translational invariance 
    self.inception1 = [ [], [], [], [], []]
    huge = self.inception1[0]
    huge.append( Conv2DPeriodic( filters=5, kernel_size=15, strides=stride, activation='relu') )
    huge.append( Conv2DPeriodic( filters=5, kernel_size=7, strides=stride, activation='relu') )
    large = self.inception1[1]
    large.append( Conv2DPeriodic( filters=5, kernel_size=11, strides=stride, activation='relu') )
    large.append( Conv2DPeriodic( filters=5, kernel_size=5, strides=stride, activation='relu') )
    medium= self.inception1[2]
    medium.append( Conv2DPeriodic( filters=5, kernel_size=5, strides=stride, activation='relu') )
    small = self.inception1[3]
    small.append( Conv2DPeriodic( filters=5, kernel_size=3, strides=stride, activation='relu') )
    layer_concat = self.inception1[-1]
    layer_concat.append( Concatenate())
    layer_concat.append( Conv2D( filters=10, kernel_size=1, strides=stride, activation='selu'))
    layer_concat.append( BatchNormalization() )


    self.cnn_wrap = []
    self.cnn_wrap.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=stride, activation='relu' ))
    self.cnn_wrap.append( BatchNormalization() )
    self.cnn_wrap.append( GlobalAveragePooling2D() )

    self.regressor = []
    self.regressor.append( Dense( 80, activation='selu' )) 
    self.regressor.append( BatchNormalization()) 
    self.regressor.append( Dense( 50, activation='selu' )) 
    self.regressor.append( BatchNormalization()) 
    self.regressor.append( Dense( 20, activation='selu' )) 
    self.regressor.append( BatchNormalization()) 
    self.regressor.append( Dense( self.n_output ) )


  def call( self, vol, images, training=False, *args, **kwargs):
    x_vol = self.predict_vol( vol, training=training)
    x_cnn = self.predict_cnn( images, training=training)
    return x_vol + x_cnn

  def predict_cnn( self, images, training=False):
    x = []
    for i in range( len( self.inception1) -1 ):
        for j in range( len( self.inception1[i]) ):
            if j == 0:
                x.append( self.inception1[i][j]( images, training=training) )
            else:
                x[i] = self.inception1[i][j]( x[i], training=training ) 
    #concatenation and 1x1 convo
    for layer in self.inception1[-1]:
        x = layer( x, training=training)
    #conv and pooling
    for layer in self.cnn_wrap:
        x = layer( x, training=training)
    # regression of fatures
    for layer in self.regressor:
        x = layer(x, training=training)
    return x



class OnlyCnn( Model):
    """ a naked convolutional neural network without anything else """
    def __init__( self, n_output):
        super().__init__( n_output)
        self.build_model()

    def build_model( self,  dense=[100,70,50,30], activation='selu', batch_norm=True, **conv_architecture ):
        """
        dense:          list like of ints, default [256,128,64,32]
                        dense layer after pooling
        activation:     str or list like of str, default 'selu'
                        activation function for dense layer
        batch_norm:     bool, default True
                        whether to apply batch normalization after dense and pooling
        **conv_architecture with default values:
        kernels:        list like of ints, default [11,7,5,3,3]
                        dimension of each kernel, len(kernels) == n_conv
        strides:        list like of ints, default [4,3,3,2,2]
                        stride of each kernel, has to match len( kernels)
        filters:        list like of ints, default [32,32,64,64,96]
                        number of channels per layer, has to match len( kernels)
        pre_pool:       int, default 2
                        how much the image should be downsampled before convolution
        pooling:        bool or list of ints, default True
                        if booling should be applied after every layer,
                        can be specified with ints, defaults to size 2 in each layer
        Returns:
        --------
        None:           allocates architecture inside
        """ 
        ## Input preprocessing
        kernels = conv_architecture.pop( 'kernels', [11,7,5,3,3])
        strides = conv_architecture.pop( 'strides', [4,3,3,2,2])
        filters = conv_architecture.pop( 'filters', [32,32,64,64,96])
        pooling = conv_architecture.pop( 'pooling', True)
        pre_pool = conv_architecture.pop( 'pre_pool', 2)
        n_conv = len( kernels)
        if isinstance( pooling, bool) and pooling is True:
            pooling = n_conv*[2]
        elif isinstance( pooling, int):
            pooling = n_conv*[pooling]
        elif pooling is False:
            pooling = n_conv*[pooling]
        ## build conv net
        conv_net = []
        if pre_pool:
            conv_net.append( AvgPool2DPeriodic( pre_pool) )
        for i in range( len(kernels)):
             conv_net.append( Conv2DPeriodic( filters=filters[i], kernel_size=kernels[i], strides=strides[i], activation=activation) )
             if pooling[i]:
                 conv_net.append( MaxPool2DPeriodic( pooling[i]) )
        conv_net.append( Flatten() )
        ## build the regressor part
        predictor = []
        if batch_norm is True:
            predictor.append( BatchNormalization() )
        for i in range( len(dense)):
            predictor.append( Dense( dense[i], activation=activation ) )
            if batch_norm is True:
                predictor.append( BatchNormalization() )
        predictor.append( Dense( self.n_output, activation=None ) )
        #put into attribute
        self.architecture.extend( conv_net)
        self.architecture.extend( predictor)


class CnnBypass( OnlyCnn, VolBypass):
  """ take the same architecture as the only cnn and add the vol bypass """
  def __init__( self, n_output, *args, **kwargs):
    super().__init__( n_output)

  def predict_cnn(self, x, training=False):
    """ take the images and predict the cnn"""
    for layer in self.architecture:
        x = layer( x, training=training) 
    return x


  def call(self, images, vol=None, training=False, *args, **kwargs):
    """ full prediction of the model, compute the volume fraction if not given"""
    if images.ndim < 4 and vol is not None:
        vol, images = images, vol
    if vol is None: 
        vol = tf.reshape( tf.reduce_mean( images, axis=[1,2,3] ), (-1, 1) )
    x_vol = self.predict_vol( vol, training=training )
    x_cnn = self.predict_cnn( images, training=training)
    return x_vol + x_cnn



class InceptionTest( Model):
  def __init__( self, n_output, n_neurons=[128,80,64,32], n_channels=[32,64,128], downsampling=4, *args, **kwargs):
    """
    try out the defined inception module from my layers, simply take a few of those
    and take some generic dense model therafter
    """
    super().__init__( n_output)
    model = self.architecture
    model.append( AvgPool2DPeriodic( downsampling) )
    ## inception modules 
    for n in n_channels:
        model.append( InceptionModule( n_out=n, n_branch=n//4) )
    model.append( Flatten() )
    ## dense regressor
    for n in n_neurons:
        model.append( Dense(n, activation='selu' ) )
        model.append( BatchNormalization() )
    model.append( Dense( n_output) )
