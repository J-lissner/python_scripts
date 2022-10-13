from tensorflow.keras.layers import Conv2D, Concatenate, concatenate, AveragePooling2D, MaxPool2D
from tensorflow.keras.layers import BatchNormalization, Dense
from tensorflow.python.ops import nn, nn_ops


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
    ## there might be a bug with 'left right' padding in even kernels, gotta check, i REALLY need to check the evaluation
    if pad_r == 0:
        top_pad = []
    else:
        top_pad = [concatenate( [data[:,-pad_r:, -pad_u:], data[:,-pad_r:,:], data[:,-pad_r:, :pad_b] ], axis=2 ) ]
    bot_pad = [concatenate( [data[:,:pad_l, -pad_u:], data[:,:pad_l,:], data[:,:pad_l, :pad_b] ], axis=2 ) ]
    data = concatenate( [data[:,:,-pad_u:], data, data[:,:,:pad_b] ], axis=2 )
    data = concatenate( top_pad + [data] + bot_pad, axis=1 )
    return data


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





















### Older slightly buggy functions which do not precisely work on padding 2
### HOWEVER, the layers have been developed with these functions and that
### affects exactly 1 subbranch layer where it is copied (see above)
def _pad_periodic( kernel_size, data):
    """
    Pad a given tensor of shape (n_batch, n_row, n_col, n_channels) 
    periodically by kernel_size//2
    This function is slightly buggy for kernel size of size 2, however
    it has been previously used
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

class _MaxPool2DPeriodic( MaxPool2D):
    """
    shadows the MaxPool2D layer from keras and implements a 
    periodic padding on runtime in each layer. All parameters except 
    the listed ones below are directly passed to the Conv2D layer and 
    follow default behaviour. CARE: This functions deploys the ever so
    slightly buggy periodic_padding
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
            return super().__call__( _pad_periodic( self.k_size, data), *args, **kwargs ) 

        else:
            raise Exception( 'requested padding "{}" is not implemented yet'.format( self.pad) )

class _AvgPool2DPeriodic( AveragePooling2D):
    """
    shadows the AveragePooling2D layer from keras and implements a 
    periodic padding on runtime in each layer. All parameters except 
    the listed ones below are directly passed to the Conv2D layer and 
    follow default behaviour. CARE: This functions deploys the ever so
    slightly buggy periodic_padding
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
            return super().__call__( _pad_periodic( self.k_size, data), *args, **kwargs ) 
        else:
            raise Exception( 'requested padding "{}" is not implemented yet'.format( self.pad) )
