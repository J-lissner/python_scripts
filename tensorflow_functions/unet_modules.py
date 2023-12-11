import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.math import ceil
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Concatenate, Add, concatenate

from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, Conv2DTransposePeriodic
from my_layers import NormalizationLayer, LayerWrapper

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
    inception_slim.append( Conv2DPeriodic( n_channels, kernel_size=3) )
    inception_slim.append( Conv2DPeriodic( n_channels, kernel_size=5) )
    self.direct_upsampler = UpSampling2D()
    self.cg_processor = LayerWrapper()
    self.cg_processor.append( inception_slim) #list inside list -> inception module
    self.cg_processor.append( Concatenate() )
    self.cg_processor.append( Conv2DPeriodic( n_channels, kernel_size=3) )
    self.cg_processor.append( Conv2DPeriodic( n_channels, kernel_size=3) )
    self.cg_processor.append( Conv2D( n_channels, kernel_size=1, activation='selu') )
    #concatenate of upsampled (features and prediction) and cg processor
    self.feature_processor = LayerWrapper( Concatenate() )
    self.feature_processor.append( NormalizationLayer( 3*n_channels + n_out) ) #bypass, up, cg_processor
    self.feature_processor.append( Conv2DPeriodic( n_channels, kernel_size=3) )
    self.feature_processor.append( Conv2DPeriodic( n_channels, kernel_size=3) )
    self.feature_processor.append( Conv2D( n_channels, kernel_size=1, activation='selu') )
    resnet_predictor = LayerWrapper( Conv2D( n_channels, kernel_size=1, activation='selu') )
    resnet_predictor.append( [Conv2DPeriodic( n_channels, kernel_size=3, strides=1, activation='selu' )] )
    resnet_predictor[-1].append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1, activation='selu' ) )
    self.side_predictor = LayerWrapper( ) #channels and cg features
    self.side_predictor.append( resnet_predictor )
    self.side_predictor.append( Add() )
    self.side_predictor.append( Conv2D( n_out, kernel_size=1, activation=None ) )

  def freeze( self, freeze=True):
    self.cg_processor.freeze( freeze)
    self.feature_processor.freeze( freeze)
    self.side_predictor.freeze( freeze)


  def __call__( self, cg_image, feature_channels, prediction=None, *layer_args, **layer_kwargs):
    """ 
    Give the models prediction on current level as well as the current 
    features in the up path
    Parameters:
    -----------
    cg_image:           tensorflow.tensor
                        coarse grained original image at current resolution
    feature_channels:   tensorflow.tensor
                        feature channels at current resolution
    prediction:         tensorflow.tensor
                        prediction of the prior level, is simply upsampled 
                        added to the prediction if given
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
    if prediction is None:
        prediction       = self.side_predictor( feature_channels, *layer_args, **layer_kwargs)
    else: #direct upsampling of the previous prediction for 'delta learning'
        prediction = (self.direct_upsampler( prediction) + 
                    self.side_predictor( [feature_channels, cg_image], *layer_args, **layer_kwargs) )
    return prediction, feature_channels


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

  def __call__( self, bypass_channels, feature_channels, prediction=None, *layer_args, **layer_kwargs):
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


class Predictor( Layer):
  def __init__( self, n_out, n_in=None, *args, **kwargs):
    """
    Parameters:
    -----------
    n_out:  int,
            number of channels to predict
    n_int:  int, default None
            number of input channels. Required for NormalizationLayer
            if given it will preceed the 1x1-SnE layer.
    """
    super().__init__( *args, **kwargs)
    self.predictor = LayerWrapper()
    if n_in is not None: 
        self.predictor.append( NormalizationLayer( n_in) )
    branch1 = Conv2D( 2*n_out, kernel_size=1)
    branch2 = [Conv2DPeriodic( 2*n_out, kernel_size=3), Conv2DPeriodic( 2*n_out, kernel_size=3) ]
    generic_resnet = LayerWrapper()
    generic_resnet.append( branch1 )
    generic_resnet.append( branch2 )
    self.predictor.append( generic_resnet)
    self.predictor.append( Add() )
    self.predictor.append( Conv2D( 2*n_out, kernel_size=1, activation='selu') )
    self.predictor.append( Conv2D( n_out, kernel_size=1, activation=None, name='final_predictor') )
    self.direct_upsampler = UpSampling2D() #only required for specific model type
  
  def freeze( self, freeze=True):
    self.predictor.freeze( freeze)

  def __call__( self, images, prediction=None, *layer_args, **layer_kwargs):
    if prediction is not None:
        prediction = self.direct_upsampler( prediction)
        return prediction + self.predictor( images)
    return self.predictor( images)

class ExperimentalPredictor( Layer):
  def __init__( self, n_out, *args, **kwargs):
    super().__init__( *args, **kwargs)
    self.predictor = LayerWrapper()
    for i in range( 1): #now with two resnet modules at last level
        branch1 = Conv2D( 2*n_out, kernel_size=1)
        branch2 = [Conv2DPeriodic( 2*n_out, kernel_size=3), Conv2DPeriodic( 2*n_out, kernel_size=3) ]
        generic_resnet = LayerWrapper()
        generic_resnet.append( branch1 )
        generic_resnet.append( branch2 )
        self.predictor.append( generic_resnet)
        self.predictor.append( Add() )
        self.predictor.append( Conv2D( 2*n_out, kernel_size=1, activation='selu') )
    self.predictor.append( Conv2D( n_out, kernel_size=1, activation=None, name='final_predictor') )
    self.direct_upsampler = UpSampling2D() #only required for specific model type
  
  def freeze( self, freeze=True):
    self.predictor.freeze( freeze)

  def __call__( self, images, prediction=None, *layer_args, **layer_kwargs):
    if prediction is not None:
        prediction = self.direct_upsampler( prediction)
        return prediction + self.predictor( images)
    return self.predictor( images)


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


class EvenDownsampler( Layer):
  """
  Module on the down path, use even sized kernels 
  also employ only add instead of concat
  """
  def __init__( self, n_channels, *args, **kwargs):
    super().__init__( *args, **kwargs)
    inception_downsampler = [ Conv2DPeriodic( n_channels, kernel_size=2, strides=2, activation='selu'  ) ]
    inception_downsampler.append(  Conv2DPeriodic( n_channels, kernel_size=4, strides=2, activation='selu'  ) )
    self.downsamplers = LayerWrapper( inception_downsampler)
    self.downsamplers.append( Add() )
    self.downsamplers.append( Conv2DPeriodic( n_channels, kernel_size=3, activation='selu') )

  def append( self, layer):
    self.downsamplers.append( layer)

  def __call__( self, images, *layer_args, **layer_kwargs):
    return self.downsamplers( images, *layer_args, **layer_kwargs)

  def freeze( self, freeze=True):
    self.downsamplers.freeze( freeze)


class FeatureTransmuter( Layer):
    """
    Collect all features (cg image, down path, up path, previous prediction)
    and gather them into a single stack, then transform the features for
    current level using two res blocks
    """
    def __init__( self, n_channels, pooling_factor, *args, **kwargs):
        """
        Parameters:
        -----------
        n_channels:     int
                        number of channels at current level, governs 
                        the number of intermediate/output chanenls
        pooling_factor: int
                        pooling factor to obtain current level from 
                        original inpyt resolution
        """
        super().__init__( *args, **kwargs)
        n_resblock = 2
        self.coarse_grainer = AvgPool2DPeriodic( pooling_factor)
        self.upsampling = UpSampling2D()
        self.down_weighter = NormalizationLayer( n_channels)
        feature_upsampler = [ Conv2DTransposePeriodic( n_channels, kernel_size=2, strides=2, activation='selu') ]
        feature_upsampler.append( Conv2DTransposePeriodic( n_channels, kernel_size=4, strides=2, activation='selu' ) )

        self.feature_upsampler = LayerWrapper( feature_upsampler)
        self.feature_upsampler.append( Add() )
        self.feature_upsampler.append( NormalizationLayer( n_channels) )
        self.feature_transmuter = LayerWrapper( NormalizationLayer( n_channels) )
        for i in range( n_resblock): 
            self.feature_transmuter.append( MyResBlock( n_channels))
            self.feature_transmuter.append( Conv2DPeriodic( n_channels, kernel_size=3, activation='selu'))
            self.feature_transmuter.append( Conv2DPeriodic( n_channels, kernel_size=3, activation='selu'))


    def __call__( self, image, down_channels, up_channels=None, prediction=None, *args, training=False, **kwargs):
        plain_data = self.coarse_grainer( image)
        if prediction is not None:
            prediction = self.upsampling( prediction, training=training)
            plain_data = concatenate( [plain_data, prediction] )
        feature_channels = self.down_weighter( down_channels, training=training )
        if up_channels is not None:
            feature_channels += self.feature_upsampler( up_channels)
        feature_channels = concatenate( [plain_data, feature_channels] )
        return self.feature_transmuter( feature_channels)


    def freeze( self, freeze=True):
        """ freeze the weights of the feature transmuter, the rest is non parametric """
        try:
            self.coarse_grainer.freeze( freeze) 
        except:
            pass #is a pooling layer
        try:
            self.upsampling.freeze( freeze) 
        except:
            pass #is a pooling layer
        self.down_weighter.freeze( freeze)
        self.feature_upsampler.freeze( freeze)
        self.feature_transmuter.freeze( freeze)


class UpsamplingConcatenate( Layer):
    def __init__( self, n_channels, *args, **kwargs):
        super().__init__( *args, **kwargs)
        feature_upsampler = [ Conv2DTransposePeriodic( n_channels, kernel_size=2, strides=2, activation='selu') ]
        feature_upsampler.append( Conv2DTransposePeriodic( n_channels, kernel_size=4, strides=2, activation='selu' ) )

        self.feature_upsampler = LayerWrapper( feature_upsampler)
        self.feature_upsampler.append( Add() )
        self.feature_upsampler.append( NormalizationLayer( n_channels) )
        self.upsampling = UpSampling2D()

    def freeze( self, freeze=True):
        """ freeze the weights of the upsampling, the rest is non parametric """
        self.feature_upsampler.freeze( freeze) 

    def __call__( self, features, prediction, images=None, training=False):
        prediction = [self.upsampling( prediction)]
        if images is not None:
            prediction.append( images)
        features = [ self.feature_upsampler( features, training=training)] #list for concat
        return concatenate( features + prediction)




class MyResBlock( Layer):
    def __init__( self, n_channels, n_conv=2, bypass=None, *args, **kwargs):
        """
        Paramters:
        ----------
        n_channels: int
                    number of output channels in total and in each operation
        n_conv:     int, default 2
                    number of 3x3 convolutions in the operating branch
        bypass:     invoked layer or None
                    operation used in bypass before adding to the result
        """
        super().__init__( *args, **kwargs)
        n_conv = 2
        self.bypass = bypass
        self.conv = LayerWrapper()
        for i in range( n_conv):
            self.conv.append( Conv2DPeriodic( n_channels, kernel_size=3, activation='selu') )

    def __call_( self, images, *args, training=False, **kwargs ):
        features = self.conv( images, *args, **kwargs, training=training )
        if self.bypass is not None:
            images = self.bypass( images)
        return features + images
    
    def freeze( self, freeze=True):
        self.conv.freeze( freeze)
        if self.bypass is not None:
            try: self.bypass.freeze( freeze)
            except: self.bypass.trainable = not freeze
        


