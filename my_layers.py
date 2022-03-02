# includes

from tensorflow.keras.layers import Conv2D, Concatenate, concatenate, AveragePooling2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.python.ops import nn, nn_ops

def pad_periodic( kernel_size, data):
    """
    Pad a given tensor of shape (n_batch, n_row, n_col, n_channels) 
    periodically by kernel_size//2
    """
    if isinstance( kernel_size, int):
        pad_w = (kernel_size )//2 
        pad_h = pad_w
    else: 
        pad_w = (kernel_size[1] )//2 
        pad_h = (kernel_size[0] )//2
    top_pad = concatenate( [data[:,-pad_w:, -pad_h:], data[:,-pad_w:,:], data[:,-pad_w:, :pad_h] ], axis=2 )
    bot_pad = concatenate( [data[:,:pad_w, -pad_h:], data[:,:pad_w,:], data[:,:pad_w, :pad_h] ], axis=2 )
    data = concatenate( [data[:,:,-pad_h:], data, data[:,:,:pad_h] ], axis=2 )
    return concatenate( [top_pad, data, bot_pad], axis=1 )


class Conv2DPeriodic( Conv2D):
    """
    shadows the Conv2D layer from keras and implements a periodic padding
    on runtime in each layer.
    Changed parameters:
    padding:    string, default same_periodic
                Padding such that the layer output either stays the same
                or is reduced (option 'valid'). Allows for same_x and
                same_y for only one directional periodicity. #TO BE IMPLEMENTED
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

    def __call__(self, data, *args, **kwargs):
        if self.pad == 'valid':
            return super().__call__( data, *args, **kwargs)
        elif self.pad in ['same', 'same_periodic']:
            return super().__call__( pad_periodic( self.k_size, data), *args, **kwargs ) 
        else:
            raise Exception( 'requested padding is not implemented yet')



class AvgPool2DPeriodic( AveragePooling2D):
    """
    shadows the AveragePooling2D layer from keras and implements a periodic padding
    on runtime in each layer.
    Changed parameters:
    padding:    string, default same_periodic
                Padding such that the layer output either stays the same
                or is reduced (option 'valid'). Allows for same_x and
                same_y for only one directional periodicity. #TO BE IMPLEMENTED
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
    shadows the MaxPool2D layer from keras and implements a periodic padding
    on runtime in each layer.
    Changed parameters:
    padding:    string, default same_periodic
                Padding such that the layer output either stays the same
                or is reduced (option 'valid'). Allows for same_x and
                same_y for only one directional periodicity. #TO BE IMPLEMENTED
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

