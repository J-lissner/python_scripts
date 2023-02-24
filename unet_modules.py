import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.math import ceil
from tensorflow.python.trackable.data_structures import ListWrapper
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Concatenate, Add, concatenate

from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, Conv2DTransposePeriodic



class LayerWrapper(ListWrapper):
  """
  This function is like a callable ListWrapper. It is used to store any
  type of layer, also inception modules followed (or preceeded) by normal
  layers. Is able to consider deep inception modules but not nested inception
  modules. 
  Normal layers are given by layer classes, Inception modules are given by
  a list of layers, and deep inception modules are given by nested lists of layers.
  """
  def __init__( self, *args, **kwargs):
      super().__init__( list(args), **kwargs)

  def __call__( self, images, *layer_args, **layer_kwargs):
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
                  additional inputs directy passed to each layer
      """
      for layer in self:
          if isinstance( layer,list): #inception module
              inception_pred = []
              for layer in layer:
                  if isinstance( layer, list): #deep inception module
                      inception_pred.append( layer[0](images, *layer_args, **layer_kwargs) )
                      for deeper_layer in layer[1:]:
                          inception_pred[-1] = deeper_layer(inception_pred[-1], *layer_args, **layer_kwargs )
                  else: #only parallel convolutions
                      pred = layer(images, *layer_args, **layer_kwargs) 
                      inception_pred.append(pred)
              images = inception_pred
          else: #simple layer, or concatenator after inception module
              images = layer( images, *layer_args, **layer_kwargs) 
      return images

  def freeze( self, freeze=True):
    for layer in self:
        if isinstance( layer, LayerWrapper):
            layer.freeze( freeze) 
        elif isinstance( layer, list):
            for sublayer in layer:
                sublayer.trainable = not freeze
        else:
            layer.trainable = not freeze


class SidePredictor( Layer):
  """
  This module predicts the solution on current resolution. For the prediction
  it requires the coarse grained image, and the feature channels of current 
  resolution
  """
  def __init__( self, n_channels, n_out, *args, **kwargs):
    """ 
    build the module
    Builds the coarse grain processor, the feature processor and the predictor
    Parameters:
    -----------
    n_channels:     int, 
                    number of channels each convolution/intermediate step should output
    n_out:          int,
                    number of solution channels 
    *args, **kwargs: input arguments passed to the parent __init__ method 
    """
    super().__init__( *args, **kwargs)
    #conv1x1 = lambda n: Conv2D( n, kernel_size=1, activation='selu' )
    inception_slim    = LayerWrapper()
    inception_slim.append( Conv2D( 1, kernel_size=1, activation=None)  ) #default stride is 1
    inception_slim.append( Conv2DPeriodic( n_channels, kernel_size=3, activation='selu') )
    inception_slim.append( Conv2DPeriodic( n_channels, kernel_size=5, activation='selu') )
    self.cg_processor = LayerWrapper()
    self.cg_processor.append( inception_slim) #list inside list -> inception module
    self.cg_processor.append( Concatenate() )
    self.cg_processor.append( Conv2D( n_channels, kernel_size=1, activation='selu' ) )
    #concatenate of upsampled (features and prediction) and cg processor
    self.feature_processor = LayerWrapper( Concatenate() )
    self.feature_processor.append( Conv2DPeriodic( n_channels, kernel_size=3) )
    self.feature_processor.append( Conv2D( n_channels, kernel_size=1, activation='selu' ) )
    inception_predictor = LayerWrapper( Conv2D( n_channels, kernel_size=1, activation='selu') )
    inception_predictor.append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1, activation='selu' ) )
    inception_predictor.append( Conv2DPeriodic( n_channels, kernel_size=5, strides=1, activation='selu' ) )
    inception_predictor.append( MaxPool2DPeriodic( 2, strides=1) )
    self.side_predictor = LayerWrapper( Concatenate() ) #channels and cg features
    self.side_predictor.append( inception_predictor )
    self.side_predictor.append( Concatenate() )
    self.side_predictor.append( Conv2D( n_out, kernel_size=1, activation=None ) )

  def freeze( self, freeze=True):
    self.cg_processor.freeze( freeze)
    self.feature_processor.freeze( freeze)
    self.side_predictor.freeze( freeze)


  def __call__( self, cg_image, feature_channels, *layer_args, **layer_kwargs):
    """ 
    Give the models prediction on current level as well as the current 
    features in the up path
    Parameters:
    -----------
    cg_image:           tensorflow.tensor
                        coarse grained original image at current resolution
    feature_channels:   tensorflow.tensor
                        feature channels at current resolution
    *layer_kw/args:     kw/args directly passed to each layer call 
    Returns:
    --------
    prediction:         tensorflow.tensor
                        prediction of current level
    level_features:     tensorflow.tensor
                        features of current level (same resolution as prediction) 
    """
    feature_channels = [feature_channels, self.cg_processor( cg_image, *layer_args, **layer_kwargs) ]
    feature_channels = self.feature_processor( feature_channels, *layer_args, **layer_kwargs)
    prediction       = self.side_predictor( [feature_channels, cg_image], *layer_args, **layer_kwargs)
    return prediction, feature_channels


class Predictor( Layer):
  def __init__( self, n_out, *args, **kwargs):
    super().__init__( *args, **kwargs)
    branch1 = [MaxPool2DPeriodic( 2, strides=1 ), Conv2D( n_out, kernel_size=1, activation='selu')]
    branch2 = [Conv2D( n_out, kernel_size=1, activation='selu'), Conv2DPeriodic( n_out, kernel_size=3, activation='selu') ]
    branch3 = [Conv2D( n_out, kernel_size=1, activation='selu'), Conv2DPeriodic( n_out, kernel_size=5, activation='selu')  ]
    generic_inception = LayerWrapper()
    generic_inception.append( branch1 )
    generic_inception.append( branch2 )
    generic_inception.append( branch3 )
    self.predictor = LayerWrapper()
    self.predictor.append( generic_inception)
    self.predictor.append( Concatenate() )
    self.predictor.append( Conv2D( n_out, kernel_size=1, activation=None, name='final_predictor') )
  
  def freeze( self, freeze=True):
    self.predictor.freeze( freeze)

  def __call__( self, images, *layer_args, **layer_kwargs):
    return self.predictor( images)


class FeatureConcatenator( Layer):
  """
  This block take care of merging all the features together, i.e. the bypass
  connection from the down path, the feature channels and the prediction from
  the lower layer.
  Also takes care of the upsampling of the lower resolution feature channels
  """
  def __init__( self, n_channels, *args, **kwargs):
    """ n_channels is here the number of channels for the next level"""
    super().__init__( *args, **kwargs)
    self.prediction_upsampler = LayerWrapper( UpSampling2D()  )
    self.feature_upsampler = LayerWrapper()
    # inception like structure
    upsampler = LayerWrapper( Conv2DTransposePeriodic( n_channels, kernel_size=2, strides=2, activation='selu', padding='same', **kwargs) )
    upsampler.append( Conv2DTransposePeriodic( n_channels, kernel_size=4, strides=2, activation='selu', padding='same', **kwargs) )
    self.feature_upsampler.append( upsampler )
    self.feature_upsampler.append( Concatenate() )
    self.feature_upsampler.append( Conv2D( n_channels, kernel_size=1, activation='selu' ) ) 

  def freeze( self, freeze=True):
    self.prediction_upsampler.freeze( freeze)
    self.feature_upsampler.freeze( freeze)

  def __call__( self, bypass_channels, feature_channels, prediction, *layer_args, **layer_kwargs):
    """
    upsample the feature channels and the previous prediction indepenently
    Then concatenate all of the three arguments.
    Parameters:
    -----------
    bypass_channels:  tensorflow.tensor
                      feature channels of the down path at current level
                      If not given, 'None' has to be passed instead
    feature_channels: tensorflow.tensor
                      feature channels of the up path at previous level
                      If not given, 'None' has to be passed instead
    prediction:       tensorflow.tensor
                      prediction of the model on previous level
                      If not given, 'None' has to be passed instead
    """
    bypass_channels  = [] if bypass_channels is None else [bypass_channels] 
    feature_channels = [] if feature_channels is None else [self.feature_upsampler( feature_channels, *layer_args, **layer_kwargs)]
    prediction       = [] if prediction is None else [self.prediction_upsampler( prediction, *layer_args, **layer_kwargs)]
    return concatenate( bypass_channels + feature_channels + prediction )

class InceptionUpsampler( Layer):
  """
  Have an inception like conv2d transpose with periodic padding
  has kernel size 2 and 4 to circumvent checkerboard pattern
  """
  def __init__( self, n_channels, *args, **kwargs): 
    super().__init__( *args, **kwargs)
    upsampler = LayerWrapper( Conv2DTransposePeriodic( n_channels, kernel_size=2, strides=2, activation='selu', padding='same', **kwargs) )
    upsampler.append( Conv2DTransposePeriodic( n_channels, kernel_size=4, strides=2, activation='selu', padding='same', **kwargs) )
    self.feature_upsampler = LayerWrapper()
    self.feature_upsampler.append( upsampler )
    self.feature_upsampler.append( Concatenate() )
    self.feature_upsampler.append( Conv2D( n_channels, kernel_size=1, activation='selu' ) ) 

  def append( self, layer):
      self.feature_upsampler.append( layer)

  def __call__( self, images, *layer_args, **layer_kwargs):
    return self.feature_upsampler( images, *layer_args, **layer_kwargs) 

  def freeze( self, freeze=True):
    self.feature_upsampler.freeze( freeze)


class InceptionEncoder( Layer):
  """
  This block take care of merging all the features together, i.e. the bypass
  connection from the down path, the feature channels and the prediction from
  the lower layer.
  Also takes care of the upsampling of the lower resolution feature channels
  """
  def __init__( self, n_channels, maxpool=True, *args, **kwargs):
    super().__init__( *args, **kwargs)
    inception_downsampler = LayerWrapper( Conv2DPeriodic( n_channels, kernel_size=3, strides=2, activation='selu'  ) )
    inception_downsampler.append(  Conv2DPeriodic( n_channels, kernel_size=5, strides=2, activation='selu'  ) )
    if maxpool:
        inception_downsampler.append( MaxPool2DPeriodic( 2))
    self.downsamplers = LayerWrapper( inception_downsampler)
    self.downsamplers.append( Concatenate() )
    self.downsamplers.append( Conv2D( n_channels, kernel_size=1, activation='selu') )

  def append( self, layer):
    self.downsamplers.append( layer)

  def __call__( self, images, *layer_args, **layer_kwargs):
    return self.downsamplers( images, *layer_args, **layer_kwargs)

  def freeze( self, freeze=True):
    self.downsamplers.freeze( freeze)



