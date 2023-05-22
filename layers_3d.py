import tensorflow as tf
from tensorflow.keras.layers import Conv3D, AveragePooling3D, MaxPool3D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import GlobalAveragePooling3D, GlobalMaxPool3D
from tensorflow.keras.layers import BatchNormalization, Dense, Concatenate, concatenate 
from tensorflow.python.ops import nn, nn_ops
from math import ceil, floor
from my_layers import LayerWrapper






#### Single layers 
class Conv3DPeriodic( Conv3D):
    """
    shadows the Conv3D layer from keras and implements a 
    periodic padding on runtime in each layer. All parameters except
    the listed ones below are directly passed to the Conv3D layer and
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



class AvgPool3DPeriodic( AveragePooling3D):
    """
    shadows the AveragePooling3D layer from keras and implements a 
    periodic padding on runtime in each layer. All parameters except 
    the listed ones below are directly passed to the Conv3D layer and 
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


class MaxPool3DPeriodic( MaxPool3D):
    """
    shadows the MaxPool3D layer from keras and implements a 
    periodic padding on runtime in each layer. All parameters except 
    the listed ones below are directly passed to the Conv3D layer and 
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
    image_dim = data.ndim - 2 #n_smaples, n_channels
    if isinstance( kernel_size, int):
        kernel_size = image_dim*[kernel_size]
    for i in range( image_dim):
        left_pad =  kernel_size[i]//2 #start of axis
        right_pad = left_pad - 1 if (kernel_size[i] %2) == 0 else left_pad #end of axis
        ## left and right refers to where its 'stitched on', i.e. slices from the opposite side
        if right_pad == 0:
            end_pad = []
        else:
            end_slice = (i+2)*[slice(None)]
            end_slice[i+1] = slice( right_pad)
            end_slice = tuple( end_slice)
            end_pad = [data[end_slice]]
        start_slice = (i+2)*[slice(None)]
        start_slice[i+1] = slice( -left_pad, data.shape[i+1])
        start_slice = tuple( start_slice)
        data = concatenate( [data[start_slice], data ] + end_pad, axis=i+1 )
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


class DeepInception( Layer):
  def __init__( self, n_channels=64, pooling='average', *args, **kwargs):
    """
    So the general idea is to have a reusable module with the deep inception modules
    I will start by just trying out a downsampling factor of 8 after each module
    With no fancy bypasses and just averagepooling and convolution
    General layout idea is to have different downsampling factors of average pooling
    and then whatever convolution operations
    Each branch has n_features channels, with a compression to n_features//2 channels before concatenation
    channels specifies if the channels should be linearly increased or constant within
    one module
    Parameters:
    -----------
    n_channels:     list of (up to 3) ints, default 64
                    number of channels defined for convolution, linearly interpolates 
                    as the number of channels in each branch for the first two ints
                    in the list. The last integer defines the number of output channels
                    after concatenation and 1x1 conv (list of len 2 is also allowed) 
    pooling:        str, default 'average'
                    pre-pooling method in each branch, choose between 'average' and 'max'
    """
    super().__init__( *args, **kwargs)
    n_out = [n_channels] if not hasattr( n_channels, '__iter__') else list( n_channels)
    n_out = 2*n_out if len( n_out) == 1 else n_out
    ## input preprocessing
    if pooling in ['average', 'avg']:
        pooling = AvgPool3DPeriodic
    elif not isinstance( pooling, Layer):
        pooling = MaxPool3DPeriodic
    self.module = LayerWrapper( [] )
    ## increasinlgy higher coarse graining branches
    default_conv = lambda n_out=n_out[1], kernel_size=3, strides=1: Conv3DPeriodic( n_out, kernel_size, strides) 
    conv_1x1     = lambda n_out=n_out[1]: Conv3D( n_out, kernel_size=1, activation='selu' )
    n_channels   = lambda i, n_conv: n_out[0] + i/n_conv * (n_out[1] - n_out[0] )
    # 6 conv, 3 1x1 
    n_op = 6 #number of actual convolution operations
    branch1 = []
    branch1.append( default_conv(  n_out=n_channels(1, n_op), strides=2) )
    branch1.append( default_conv(  n_out=n_channels(2, n_op)) )
    branch1.append( conv_1x1(      n_out=n_channels(2, n_op)) )
    branch1.append( default_conv(  n_out=n_channels(3, n_op), strides=2) )
    branch1.append( default_conv(  n_out=n_channels(4, n_op)) )
    branch1.append( conv_1x1(      n_out=n_channels(4, n_op)) )
    branch1.append( default_conv(  n_out=n_channels(5, n_op), strides=2) )
    branch1.append( SqueezeExcite( n_channels(6, n_op), default_conv( n_out=n_channels(6, n_op))) )
    #branch1.append( default_conv()) 
    #pool+ 1 1x1 + 4 conv operations
    n_op = 4
    branch2 = [pooling( 2)] 
    branch2.append( default_conv(  n_out=n_channels(1, n_op), kernel_size=5, strides=2) )
    branch2.append( default_conv(  n_out=n_channels(2, n_op)) )
    branch2.append( conv_1x1(      n_out=n_channels(2, n_op)) )
    branch2.append( default_conv(  n_out=n_channels(3, n_op), kernel_size=5, strides=2) )
    branch2.append( SqueezeExcite( n_channels(4, n_op), default_conv( n_out=n_channels(4, n_op)) ) )
    #branch2.append( default_conv() ) 
    #pool + 1 1x1 + 3conv
    n_op =  3
    branch3 = [pooling( 4)]
    branch3.append( default_conv(  n_out=n_channels(1, n_op), kernel_size=5, strides=2) )
    branch3.append( default_conv(  n_out=n_channels(2, n_op), kernel_size=5) )
    branch3.append( conv_1x1(      n_out=n_channels(2, n_op) ) )
    branch3.append( SqueezeExcite( n_channels(3, n_op), default_conv( n_out=n_channels(3, n_op)) ) )
    #branch3.append( default_conv( n_out=n_channels(3, n_op)) ) 
    #pool + 1 1x1 + 3 conv
    n_op = 3
    branch4 = [pooling( 8)]
    branch4.append( default_conv(  n_out=n_channels(1, n_op), kernel_size=5 ) )
    branch4.append( default_conv(  n_out=n_channels(2, n_op), kernel_size=5 ) )
    branch4.append( conv_1x1(      n_out=n_channels(2, n_op) ) )
    branch4.append( SqueezeExcite( n_channels(3, n_op), default_conv( n_out=n_channels(3, n_op), kernel_size=5 )  ) )
    #branch4.append( default_conv( n_out=n_channels(3, n_op), kernel_size=5 )  ) 
    self.module[-1].extend( [branch1, branch2, branch3, branch4])
    self.module.append( Concatenate() )
    self.module.append( SqueezeExcite( n_out[-1], Conv3D( n_out[-1], kernel_size=1, activation='selu' )) )



  def freeze( self, freeze=True):
      self.module.freeze( freeze)

  def __call__( self, images, x_extra=None, training=False):
      """
      Predict the deep inception module. May also take an extra feature
      vector which will be (i think) added to the squeeze and excitation bypass
      Parameters:
      -----------
      images:       tensorflow.tensor of shape [n, *res, n_channels]
                    input data to the module
      x_extra:      tensorflow.tensor of shape [n, n_fts]
                    feature vector to be added to the SnE block
      training:     bool, default False
                    flag to inform the ANN whether its training
      Returns:
      --------
      images:       tensorflow.tensor of shape [n, *res, n_channels]
                    output of the module, downsampled by factor 8
      """
      return self.module( images, training=training)
      #return self.module( images, x_extra=x_extra, training=training)

class SqueezeExcite(Layer):
    """
    get the Squeeze and excitation block for any layer, or callable. Only
    works in the 3D convolutional neural netowrk context (since globalPool3D
    is used)
    It is basically a reweighting of channels using the global information, as
    well as each channel for information input. The main difference to 1x1 conv 
    is that it returns channel wise weights from 0-1 (sigmoid), instead of a
    sum of all channels in each channel. So it weights each individual output
    channel instead of weighting each individual input channel over all output channels
    This has added functionality to the literature squeeze and excitation 
    module, namely there can be a layer in parralel to the squeeze and excitation
    sidepass
    """
    def __init__( self, n_channels, layer=None, reduction_ratio=16, *args, **kwargs):
        """
        Implement the defaut squeeze and excitation network
        Parameters:
        -----------
        n_channels:     int or list of ints
                        [(input), output ] n_channels of the layer, if an 
                        int it assumes that input == output
        layer:          tensorflow.keras.layer or callable
                        callable which to squeeze and excite
        reduction_rate: int, default 16
                        reduction rate of the channels 
        """
        ## input processing
        super().__init__( *args, **kwargs)
        n_channels = [n_channels] if not hasattr( n_channels, '__iter__') else list( n_channels)
        n_channels = 2*n_channels if len( n_channels) == 1 else n_channels
        layer = (lambda x, *args, **kwargs: x)  if layer is None else layer
        self.pooling = GlobalAveragePooling3D()
        self.se_block = LayerWrapper()
        self.se_block.append( Dense( n_channels[0]//reduction_ratio, activation='selu') )
        self.se_block.append( Dense( n_channels[1], activation='sigmoid' ))
        self.layer = layer

    def __call__( self, images, *channel_args, x_extra=None, training=False, **channel_kwargs):
        weights = self.pooling( images)
        if x_extra is not None:
            weights = concatenate( [weights, x_extra] )
        weights = self.se_block( weights, training=training)
        weights = tf.reshape( weights, [weights.shape[0], 1,1,1, weights.shape[-1] ] )
        return weights * self.layer( images, *channel_args, training=training, **channel_kwargs) 

    def freeze( self, freeze=False):
        self.se_block.freeze( freeze)
        try: 
            self.layer.freeze( freeze) #layerwrapper
        except:
            self.layer.trainable = not freeze


class NormalizationLayer( SqueezeExcite):
    """
    basically the same as the Squeeze and Excite block above, but there is a
    default 1x1 conv at the same time to the SnE bypass
    """
    def __init__( self, n_channels, reduction_ratio=16, *args, **kwargs):
        """
        Implement the squeeze and excitation block with parallel 1x1 conv
        Parameters:
        see SqueezeExcite.__init__ 
        """
        n_channels = 2*[n_channels] if isinstance( n_channels,int) else n_channels
        bypass_layer = Conv3D( n_channels[1], kernel_size=1, activation='selu' )
        super().__init__( n_channels, layer=bypass_layer, reduction_ratio=reduction_ratio, *args, **kwargs)


class NormalizationLayer( SqueezeExcite):
    """
    basically the same as the Squeeze and Excite block above, but there is a
    default 1x1 conv at the same time to the SnE bypass
    """
    def __init__( self, n_channels, reduction_ratio=16, *args, **kwargs):
        """
        Implement the squeeze and excitation block with parallel 1x1 conv
        Parameters:
        see SqueezeExcite.__init__ 
        """
        n_channels = 2*[n_channels] if isinstance( n_channels,int) else n_channels
        bypass_layer = Conv3D( n_channels[1], kernel_size=1, activation='selu' )
        super().__init__( n_channels, layer=bypass_layer, reduction_ratio=reduction_ratio, *args, **kwargs)



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
        bypass.append( AvgPool3DPeriodic( downsample)  )
    elif 'max' in pooling:
        bypass.append( MaxPool3DPeriodic( downsample)  )
    bypass.append( Conv3D( filters=n_branch, kernel_size=1, strides=1, activation=activation) )
    concat.append( Conv3D( filters=n_out, kernel_size=1, strides=1, activation=activation) )
    concat.append( BatchNormalization() )
    ## different convolution branches
    for i in range( n_conv):
        self.inception.append( [Conv3DPeriodic( filters=n_branch, kernel_size=3, strides=min(downsample,2), activation=activation ) ])
        k = 1
        while 2*k/downsample < 1 or k < i:
          kernel_size = min(k,i)*2 + 3 
          if downsample/(k*2) > 1:
              stride = 2
          else: 
              stride = 1
          self.inception[i].append( Conv3DPeriodic( filters=n_branch, kernel_size=kernel_size, strides=stride, activation=activation ) )
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

