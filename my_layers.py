# includes

from tensorflow.keras.layers import Conv2D, concatenate, AveragePooling2D, MaxPool2D
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

    def __call__(self, data):
        if self.pad == 'valid':
            return super().__call__( data)
        elif self.pad in ['same', 'same_periodic']:
            return super().__call__( pad_periodic( self.k_size, data) ) 
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


    def __call__(self, data):
        if self.pad == 'valid':
            return super().__call__( data)
        elif self.pad in ['same', 'same_periodic']:
            return super().__call__( pad_periodic( self.k_size, data) ) 
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


    def __call__(self, data):
        if self.pad == 'valid':
            return super().__call__( data)
        elif self.pad in ['same', 'same_periodic']:
            return super().__call__( pad_periodic( self.k_size, data) ) 
        else:
            raise Exception( 'requested padding "{}" is not implemented yet'.format( self.pad) )


