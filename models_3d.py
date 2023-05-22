import tensorflow as tf
import numpy as np
import itertools
from tensorflow.math import ceil
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, Flatten, Concatenate
from conv_layers_old import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic
from layers_3d import InceptionModule, DeepInception, LayerWrapper #, Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, 
from hybrid_models import VolBypass, FeaturePredictor


##################### Convolutional Neural Networks ##################### 
class ModularInception( VolBypass): 
  def __init__( self, n_output, n_channels=64, dense=[50,32,16], module_kwargs=dict(), *args, **kwargs):
      """
      Build a very modal deep inception model which takes 2 (new) modular
      deep inception block in a consecutive manner and has a dense block
      thereafter
      Parameters:
      -----------
      n_output:     int
                    number of neurons to predict
      n_channels:   int, default 32
                    number of output channels per deep inception module
                    scales with $n$ for the $n$-th deep inception block
      dense:        list of ints, default [32,32,16,16]
                    basically number of neurons in the dense part, always
                    uses selu and batch normalization 
      """
      module_kwargs = module_kwargs.copy()
      n_channel1 = module_kwargs.pop( 'n_channel1', [0, n_channels] )
      n_channel2 = module_kwargs.pop( 'n_channel2', n_channels )

      super().__init__( n_output, *args, **kwargs)
      self.conv =  [DeepInception( n_channels=n_channel1, **module_kwargs), 
                    DeepInception( n_channels=n_channel2, pooling='max', **module_kwargs)  ]
      if len( self.conv) == 1:
        self.conv.append( GlobalAveragePooling2D() )
      else:
        self.conv.append( Flatten() )
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
          x_extra = tf.reshape( features[:,1], (-1,1) ) 
      for layer in self.conv:
          images = layer( images, training=training) #x_extra=x_extra, training=training)
      if x_extra is not None:
          #pass  #TODO commented out for debuggin purposes, such
          ## that i can see if the prediction changes  when i use different phase contrast
          images = concatenate( [images, x_extra]) #TODO
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



