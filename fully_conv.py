import tensorflow as tf
import numpy as np
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
    down_layers = lambda n_channels: Conv2DPeriodic( n_channels, kernel_size=3, strides=2, activation='selu' )
    up_layers = lambda n_channels, kernel_size: Conv2DTranspose( n_channels, kernel_size=kernel_size, strides=2, activation='selu', padding='same' )
    conv_1x1 = lambda n_channels: Conv2D( n_channels, kernel_size=1, strides=1, activation='selu' )
    for i in range( self.n_levels): #from top to bottom
        self.down_path.append( [down_layers( (i+1)*channel_per_down )] )
        #possible additional operations # self.down_path[-1].append()
    ## now the path up, take averagepooling + 5x5 convolution and concatenate channels 
    ## i could also write a layer class which has this, and does the operation below
    ## i think that is more elegant, makes the call less cluttered indeed
    for i in range( self.n_levels, 0, -1): #from bottom to top
        self.processors.append( [AvgPool2DPeriodic( 2**(i) )] )  #convolutions on each level to the right
        self.processors[-1].append( Conv2DPeriodic( i*channel_per_down, kernel_size=5, strides=1) )
        ## possible additional  operations #self.processors[-1].append()
        ### concatenators and channel reduction before upsampling
        self.concatenators.append( [Concatenate()] )
        self.concatenators[-1].append( conv_1x1( i*channel_per_down) )
        ## possible addition of operations
        self.concatenators[-1].append( Concatenate() )
        ### upsampling layers, parallel passes with 1x1
        #inception like structure
        self.upsamplers.append( [[up_layers( (i)*channel_per_down, 3),up_layers( (i)*channel_per_down, 5)]] )
        self.upsamplers[-1].append( Concatenate() )
        self.upsamplers[-1].append( conv_1x1( i*channel_per_down ) ) 
        ## side out pass on each level for loss prediction
        self.side_predictors.append( [ Concatenate()] )
        self.side_predictors[-1].append( conv_1x1( channels_out ) )
    self.build_predictor( channels_out)
  

  def build_predictor( self, channels_out):
    conv_1x1 = lambda n_channels: Conv2D( n_channels, kernel_size=1, strides=1, activation='selu' )
    channels = max( 1, 2*channels_out//3 )
    self.predictor = []
    self.predictor.append( Concatenate() )
    self.predictor.append( [conv_1x1( channels) ] )
    self.predictor[-1].append( Conv2DPeriodic( channels, kernel_size=3, strides=1)  ) 
    self.predictor[-1].append( Conv2DPeriodic( channels, kernel_size=5, strides=1)  ) 
    #final prediction
    self.predictor.append( Concatenate() )
    self.predictor.append( conv_1x1( channels_out) )
    ## also i need conv2d transpose here



  def go_down( self, images, training=False):
    levels = [images] #store the image easily accessible, only reference
    for i in range( self.n_levels):
        j = 0
        for layer in self.down_path[i]:  #do the operations on each level
          if j == 0: 
            levels.insert( 0, layer( levels[0], training=training) )
          else:
            levels[0] = layer( levels[0], training=training )
          j += 1
    return levels

  def go_up( self, levels, training=False, upscaling=None):
    """
    Upscale the prediction and concatenate each of the layers.
    If training is true then a list of predictions is returned matching
    the number of levels.
    Parameters:
    -----------
    levels:     list of tf.Tensor like
                a list of all channels stored during the path down
    training:   bool, default False
                whether we are currently training, also the switch to
                predict and return the intermediate layers
    upscaling:  int, default None -> self.n_levels
                up to which level to upscale the prediction, if the number
                is smaller than <self.n_levels>, then the final prediction is
                not given. Required for pretraining.
    Returns:
    --------
    upscaled_channels:  tf.Tensor or list of tf.Tensors
                        predictions on the last or each layer, depending on parameters 
    """
    upscaling = self.n_levels if upscaling is None else upscaling
    multistage_predictions = [] #used during training
    previous_channels = []
    for i in range( self.n_levels ):
      coarse_grained = self.processors[i][0](levels[-1] )  #original image
      layer_channels = coarse_grained
      for layer in self.processors[i][1:]: #operate on the coarse grained image
          layer_channels = layer( layer_channels, training=training )  
      #concatenate down_path and current up path
      layer_channels = self.concatenators[i][0]( [layer_channels, levels[i] ] + previous_channels )
      ## do an arbitrary amount of operations in there
      for layer in self.concatenators[i][1:-1]: 
          layer_channels = layer( layer_channels, training=training )
      previous_channels = self.concatenators[i][-1]( [layer_channels, coarse_grained]) #add the coarse grained image
      ## do the conv2d transpose thingy
      for layer in self.upsamplers[i]:
          if isinstance( layer,list):
              inception_pred = []
              for layer in layer:
                  inception_pred.append( layer(layer_channels, training=training) )
              del layer_channels
              layer_channels = inception_pred
          else:
              layer_channels = layer( layer_channels, training=training )
      previous_channels = [layer_channels]  #needed as list for concatenation
      ### if we are currently training we require each of the levels to be predicted
      if training:
          multistage_predictions.append( self.side_predictors[i][0](previous_channels) )
          for layer in self.side_predictors[i][1:]:
            multistage_predictions[-1] = ( layer(multistage_predictions[-1]) )
      del coarse_grained, layer_channels
    if training:
        multistage_predictions.append( previous_channels )
        return multistage_predictions
    else:
        return previous_channels


  def predict( self, images, feature_channels, training=False):
    #concat image and channels
    prediction = self.predictor[0]( feature_channels + [images])
    for layer in self.predictor[1:]:
        if isinstance( layer,list):
            inception_pred = []
            for layer in layer:
                inception_pred.append( layer(prediction, training=training ) )
            del prediction
            prediction = inception_pred
        else:
            prediction = layer( prediction, training=training) 
    return prediction


  def call( self, images, training=False, multisage_losses=False, *args, **kwargs):
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
    multistage_losses:  bool, default False
                whether to give return prediction on each level of 
                upsampling. CAREFUL: Requires the 'training' to be not
                false, i.e. True or None.
    Returns:
    --------
    prediction: tensorflow.tensor or list of tf.tensors
                image data prediction of original resolution with
                self.n_output channels, or a list of predictions when
                multistage losses is set true.
    """
    predictions = self.go_down( images, training=training)
    predictions = self.go_up( predictions, training=training)
    if training:
        predictions[-1] = self.predict( images, predictions[-1], training=training ) 
    else:
        predictions = self.predict( images, predictions, training=training )
    return predictions
      

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




