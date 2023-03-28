from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPool2D
from tensorflow.keras.layers import Conv2DTranspose, Layer
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import BatchNormalization, Dense, Concatenate, concatenate 
from tensorflow.python.ops import nn, nn_ops
from tensorflow.python.trackable.data_structures import ListWrapper
from math import ceil, floor



class LayerWrapper(ListWrapper):
  """
  This function is like a callable ListWrapper. It is used to store any
  type  of layer(of deep inception modules and generic layers).
  Each entry in the list (of the ListWrapper) is treated as one module. This
  module may contain a generic 'Layer', or another list, which then specifies
  the custom modules, i.e. resnet/inception modules. (which is a list of 
  layers/lists, each entry/sublist specifying a 'branch')
  It is able to consider deep inception modules but not nested inception modules. 
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
      images:     tensorflow.tensor like
                  image data of at least 4 dimensions
      *layer_kw/args: 
                  additional inputs directy passed to each layer
      """
      for layer in self:
          if not isinstance( layer,list): #simple layer, or concatenator after inception module
              images = layer( images, *layer_args, **layer_kwargs) 
          else: #parallel conlvolution operations
              inception_pred = []
              for layer in layer:
                  if not isinstance( layer, list): #only single parallel convolutions
                      pred = layer(images, *layer_args, **layer_kwargs) 
                      inception_pred.append(pred)
                  else: #deep inception module
                      inception_pred.append( layer[0](images, *layer_args, **layer_kwargs) )
                      for deeper_layer in layer[1:]:
                          inception_pred[-1] = deeper_layer(inception_pred[-1], *layer_args, **layer_kwargs )
              images = inception_pred
      return images

  def freeze( self, freeze=True):
    for layer in self:
        if isinstance( layer, LayerWrapper):
            layer.freeze( freeze) 
        elif isinstance( layer, list):
            for sublayer in layer:
              if isinstance( sublayer, list):
                for subsublayer in sublayer:
                  subsublayer.trainable = not freeze
              else:
                sublayer.trainable = not freeze
        else:
            layer.trainable = not freeze




#### Single layers 
class Conv2DPeriodic( Conv2D):
    """
    shadows the Conv2D layer from keras and implements a 
    periodic padding on runtime in each layer. All parameters except
    the listed ones below are directly passed to the Conv2D layer and
    follow default behaviour
    Changed parameters:
    padding:    string, default same_periodic
                Padding such that the layer output either stays the same
                or is reduced (option 'valid').
                Note that 'same' is equivalent to 'same_periodic'. This layer
                does not allow 0 padding
                Also note that padding has to be set by kwargs, not by args
    """
    def __init__( self, *args, **kwargs):
        if 'padding' in kwargs:
            self.pad = kwargs['padding'].lower()
        else:
            self.pad = 'same'
        kwargs['padding'] = 'valid' #overwrite to remove any padding on operation
        super().__init__( *args, **kwargs)
        if 'kernel_size' in kwargs:
            self.k_size = kwargs['kernel_size']
        else: 
            self.k_size = args[1]
        if 'dilation_rate' in kwargs:
            self.k_size =  self.k_size + (self.k_size -1) * (kwargs['dilation_rate']-1)

    def __call__(self, data, *args, **kwargs):
        if self.pad == 'valid':
            return super().__call__( data, *args, **kwargs)
        elif self.pad in ['same', 'same_periodic']:
            return super().__call__( pad_periodic( self.k_size, data), *args, **kwargs ) 
        else:
            raise Exception( 'requested padding is not implemented yet')



class AvgPool2DPeriodic( AveragePooling2D):
    """
    shadows the AveragePooling2D layer from keras and implements a 
    periodic padding on runtime in each layer. All parameters except 
    the listed ones below are directly passed to the Conv2D layer and 
    follow default behaviour
    Changed parameters:
    padding:    string, default same_periodic
                Padding such that the layer output either stays the same
                or is reduced (option 'valid').
                Note that 'same' is equivalent to 'same_periodic'. This layer
                does not allow 0 padding
                Also note that padding has to be set by kwargs, not by args
    """
    def __init__( self, *args, **kwargs):
        if 'padding' in kwargs:
            self.pad = kwargs['padding'].lower()
        else:
            self.pad = 'same'
        kwargs['padding'] = 'valid' #overwrite to remove any padding on operation
        super().__init__( *args, **kwargs)
        if 'pool_size' in kwargs:
            self.k_size = kwargs['pool_size']
        else: 
            self.k_size = args[0]

    def __call__(self, data, *args, **kwargs):
        if self.pad == 'valid':
            return super().__call__( data, *args, **kwargs)
        elif self.pad in ['same', 'same_periodic']:
            return super().__call__( pad_periodic( self.k_size, data), *args, **kwargs ) 
        else:
            raise Exception( 'requested padding "{}" is not implemented yet'.format( self.pad) )


class MaxPool2DPeriodic( MaxPool2D):
    """
    shadows the MaxPool2D layer from keras and implements a 
    periodic padding on runtime in each layer. All parameters except 
    the listed ones below are directly passed to the Conv2D layer and 
    follow default behaviour
    Changed parameters:
    padding:    string, default same_periodic
                Padding such that the layer output either stays the same
                or is reduced (option 'valid').
                Note that 'same' is equivalent to 'same_periodic'. This layer
                does not allow 0 padding
                Also note that padding has to be set by kwargs, not by args
    """
    def __init__( self, *args, **kwargs):
        if 'padding' in kwargs:
            self.pad = kwargs['padding'].lower()
        else:
            self.pad = 'same'
        kwargs['padding'] = 'valid' #overwrite to remove any padding on operation
        super().__init__( *args, **kwargs)
        if 'pool_size' in kwargs:
            self.k_size = kwargs['pool_size']
        else: 
            self.k_size = args[0]

    def __call__(self, data, *args, **kwargs):
        if self.pad == 'valid':
            return super().__call__( data, *args, **kwargs)
        elif self.pad in ['same', 'same_periodic']:
            return super().__call__( pad_periodic( self.k_size, data), *args, **kwargs ) 
        else:
            raise Exception( 'requested padding "{}" is not implemented yet'.format( self.pad) )


class Conv2DTransposePeriodic( Conv2DTranspose):
    def __init__( self, *args, **kwargs):
        kwargs['padding'] = 'same' #enforce no padding and do the stuff by myself
        super(Conv2DTransposePeriodic, self).__init__( *args, **kwargs)
        if 'kernel_size' in kwargs:
            kernel_size = kwargs['kernel_size']
        else: 
            kernel_size = args[1]
        if 'strides' in kwargs:
            stride = kwargs['strides']
        else: 
            stride = args[2] 
        self.p_size = kernel_size//stride #might be a little excessive

    def __call__(self, data, *args, **kwargs):
        """
        shadow the original call, simply pad before upsampling and then 
        cut away the excessive part
        """
        upsampled = super().__call__( upsampling_padding( self.p_size, data ), *args, **kwargs)
        idx = max(2*self.p_size, 2)
        return upsampled[:, idx:-idx, idx:-idx ]


#### required functions (padding for now
def pad_periodic( kernel_size, data):
    """
    Pad a given tensor of shape (n_batch, n_row, n_col, n_channels) 
    periodically by kernel_size//2
    Parameters:
    -----------
    kernel_size:    list of ints or int
                    size of the (quadratic) kernel
    data:           tensorflow.tensor
                    image data of at least 3 channels, n_sample x n_x x n_y
    Returns:
    --------
    padded_data:    tensorflow.tensor
                    the padded data by the minimum required width
    """
    ## for now i assume that on even sized kernels the result is in the top left
    if isinstance( kernel_size, int):
        pad_l = kernel_size//2 #left
        pad_u = kernel_size//2 #up
        pad_r = pad_l - 1 if (kernel_size %2) == 0 else pad_l #right
        pad_b = pad_u - 1 if (kernel_size %2) == 0 else pad_u #bot
    else: 
        pad_l = (kernel_size[1] )//2  #left
        pad_u = (kernel_size[0] )//2 #bot
        pad_r = pad_l - 1 if (kernel_size[1] %2) == 0 else pad_l #right
        pad_b = pad_u - 1 if (kernel_size[0] %2) == 0 else pad_u #bot
    ## the subscript refer to where it is sliced off, i.e. placedo n the opposite side
    if pad_r == 0:
        top_pad = []
    else:
        top_pad = [concatenate( [data[:,-pad_r:, -pad_u:], data[:,-pad_r:,:], data[:,-pad_r:, :pad_b] ], axis=2 ) ]
    bot_pad = [concatenate( [data[:,:pad_l, -pad_u:], data[:,:pad_l,:], data[:,:pad_l, :pad_b] ], axis=2 ) ]
    data = concatenate( [data[:,:,-pad_u:], data, data[:,:,:pad_b] ], axis=2 )
    data = concatenate( top_pad + [data] + bot_pad, axis=1 )
    return data


def upsampling_padding( pad_size, data):
    """
    periodically pad the image before upsampling to enforce
    periodic adjacency correctness
    Parameters:
    -----------
    pad_size:       list of ints or int
                    size of the padding in each dimension
    data:           tensorflow.tensor
                    image data of at least 3 channels, n_sample x n_x x n_y
    Returns:
    --------
    padded_data:    tensorflow.tensor
                    the padded data by the minimum required width
    """
    ## for now i assume that on even sized kernels the result is in the top left
    if isinstance( pad_size, int):
        pad_size = 2*[pad_size]
    pad_l = pad_size[1]
    pad_u = pad_size[0]
    pad_r = pad_l
    pad_b = pad_u
    ## the subscript refer to where it is sliced off, i.e. placedo n the opposite side
    if pad_r == 0:
        top_pad = []
    else:
        top_pad = [concatenate( [data[:,-pad_r:, -pad_u:], data[:,-pad_r:,:], data[:,-pad_r:, :pad_b] ], axis=2 ) ]
    bot_pad = [concatenate( [data[:,:pad_l, -pad_u:], data[:,:pad_l,:], data[:,:pad_l, :pad_b] ], axis=2 ) ]
    data = concatenate( [data[:,:,-pad_u:], data, data[:,:,:pad_b] ], axis=2 )
    data = concatenate( top_pad + [data] + bot_pad, axis=1 )
    return data


### modules
class DeepInception( Layer):
  def __init__( self, n_features, pre_pooling=[2,4,8,10,20], post_pooling=['max'], n_conv=3, n_vol=None, *args, **kwargs):
    """
    So the general idea is to have a reusable module with the deep inception modules
    I will start by just trying out a downsampling factor of 8 after each module
    With no fancy bypasses and just averagepooling and convolution
    General layout idea is to have different downsampling factors of average pooling
    and then whatever convolution operations
    Each branch has n_features channels, with a compression to n_features//2 channels before concatenation
    """
    super().__init__( *args, **kwargs)
    ## input preprocessing
    for i in range( len( post_pooling)):
        if isinstance( post_pooling[i], str) and 'max' in post_pooling[i].lower(): 
            post_pooling[i] = GlobalMaxPool2D
        elif isinstance( post_pooling[i], str) and 'av' in post_pooling[i].lower(): 
            post_pooling[i] = GlobalAveragePooling2D
    n_out = n_features // len( post_pooling)
    if len( post_pooling) > 1:
        post_pooling = LayerWrapper( [ post_pooling[0](), post_pooling[1]()]  ) #invoke layers and concat
        post_pooling.append( Concatenate() )
    else:
        post_pooling = post_pooling[0]() #invoke layers
    ## variable allocation
    conv_3x3 = lambda strides=1: Conv2DPeriodic( n_features, kernel_size=3, strides=strides, activation='selu')
    self.module = LayerWrapper( [] )
    ## generate layers
    for pool in pre_pooling:
        branch = []
        if pool > 0:
            branch.append( AvgPool2DPeriodic( pool))
        for _ in range( n_conv):
            branch.append( conv_3x3() )
        branch.append( Conv2D( n_out, kernel_size=1, activation='selu' ) )
        branch.append( post_pooling ) #either a layer or a LayerWrapper, so a layer
        self.module[-1].append( branch )
    self.module.append( Concatenate() )

  def freeze( self, freeze=True):
      self.module.freeze( freeze)

  def __call__( self, images, training=False):
      """
      args to catch the modality of the code for the hybrid models
      """
      return self.module( images, training=training)


class InceptionModule():
  def __init__( self, n_out=32, n_branch=12, downsample=4, max_kernel=5, pooling='average', activation='selu'):
    """
    Define an inception module which has a 1x1 bypass and multiple two deep
    convolutional branches. The number of conovlutional branches is computed
    by <(max_kerrnel-1)//2>, since it takes them of a 2 increment.
    The inception module achieves downsampling through stride and <pooling>
    the 1x1 bypass. Always takes the periodically padded layers.
    Parameters:
    -----------
    n_out:          int, default 32
                    number of output channels of the inception module
    n_branch:       int, default 12
                    number of channels per branch in inception module
    downsample:     int, default 4
                    downsampling, must be power of 2
    max_kernel:     int, default 5
                    maximum size of the second kernel in the layer
                    preferably an odd number
    pooling:        string, default 'average'
                    max or average pooling conducted before the 1x1 bypass
    activation:     string, default 'selu'
                    activation function at each convolutional kernel
    """
    ## input preprocessing and variable allocation
    n_conv = (max_kernel-1)//2
    self.inception = []
    ## 1x1 bypass and concatenation
    bypass = []
    concat = [Concatenate()]
    if 'average' in pooling:
        bypass.append( AvgPool2DPeriodic( downsample)  )
    elif 'max' in pooling:
        bypass.append( MaxPool2DPeriodic( downsample)  )
    bypass.append( Conv2D( filters=n_branch, kernel_size=1, strides=1, activation=activation) )
    concat.append( Conv2D( filters=n_out, kernel_size=1, strides=1, activation=activation) )
    concat.append( BatchNormalization() )
    ## different convolution branches
    for i in range( n_conv):
        self.inception.append( [Conv2DPeriodic( filters=n_branch, kernel_size=3, strides=min(downsample,2), activation=activation ) ])
        k = 1
        while 2*k/downsample < 1 or k < i:
          kernel_size = min(k,i)*2 + 3 
          if downsample/(k*2) > 1:
              stride = 2
          else: 
              stride = 1
          self.inception[i].append( Conv2DPeriodic( filters=n_branch, kernel_size=kernel_size, strides=stride, activation=activation ) )
          k += 1
    ## concatenation and 1x1 convolution 
    self.inception.insert( 0, bypass)
    self.inception.append( concat )


  def __call__( self, images, training=False, *args, **kwargs):
    """ 
    evaluate the inception module, simply evaluate each branch and
    keep the output of each branch in memory until concatenated 
    """
    x = []
    for i in range( len( self.inception) -1 ):
        for j in range( len( self.inception[i]) ):
            if j == 0:
                x.append( self.inception[i][j]( images, training=training) )
            else:
                x[i] = self.inception[i][j]( x[i], training=training ) 
    #concatenation and 1x1 convo
    for layer in self.inception[-1]:
        x = layer( x, training=training)
    return x

