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
    self.depth = n_levels #how many times downsample
    channel_per_down = 4 #how many added channels per downsampling in direct convolution
    self.down_path = []
    self.up_path = []
    self.concatenators = []
    self.upsamplers = []
    ## direct path down
    down_layers = lambda n_channels: Conv2DPeriodic( n_channels, kernel_size=3, strides=2, activation='selu' )
    up_layers = lambda n_channels, kernel_size: Conv2DTranspose( n_channels, kernel_size=kernel_size, strides=2, activation='selu', padding='same' )
    conv_1x1 = lambda n_channels: Conv2D( n_channels, kernel_size=1, strides=1, activation='selu' )
    for i in range( self.depth):
        self.down_path.append( [down_layers( (i+1)*channel_per_down )] )
        #possible additional operations # self.down_path[-1].append()
    ## now the path up, take averagepooling + 5x5 convolution and concatenate channels 
    ## i could also write a layer class which has this, and does the operation below
    ## i think that is more elegant, makes the call less cluttered indeed
    for i in range( self.depth, 0, -1):
        self.up_path.append( [AvgPool2DPeriodic( 2**(i) )] ) 
        self.up_path[-1].append( Conv2DPeriodic( i*channel_per_down, kernel_size=5, strides=1) )
        ## possible additional  operations #self.up_path[-1].append()
        ### concatenators and dimensionality reduction before upsampling
        self.concatenators.append( [Concatenate()] )
        self.concatenators[-1].append( conv_1x1( i*channel_per_down) )
        ## possible addition of operations
        self.concatenators[-1].append( Concatenate() )
        ### upsampling layers, parallel passes with 1x1
        #inception like structure
        self.upsamplers.append( [[up_layers( (i)*channel_per_down, 3),up_layers( (i)*channel_per_down, 5)]] )
        self.upsamplers[-1].append( Concatenate() )
        self.upsamplers[-1].append( conv_1x1( i*channel_per_down ) ) 
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
    for operations in self.down_path: # go down all n levels
        i = 0
        for layer in operations:  #do the operations on each level
          if i == 0: 
            levels.append( layer( levels[-1], training=training) )
          else:
            levels[-1] = layer( levels[-1], training=training )
          i += 1
    return levels

  def go_up( self, levels, training=False):
    previous_channels = []
    for i in range( self.depth ):
      coarse_grained = self.up_path[i][0](levels[0] )  #original image
      layer_channels = coarse_grained
      for layer in self.up_path[i][1:]: #operate on the coarse grained image
          layer_channels = layer( layer_channels, training=training )  
      #concatenate down_path and current up path
      layer_channels = self.concatenators[i][0]( [layer_channels, levels[-i-1] ] + previous_channels )
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
      del coarse_grained, layer_channels
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

  def call( self, images, training=False, *args, **kwargs):
    predictions = self.go_down( images, training=training)
    predictions = self.go_up( predictions, training=training)
    predictions = self.predict( images, predictions, training=training )
    return predictions
      





