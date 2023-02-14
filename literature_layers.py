import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.activations import relu, selu
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, Concatenate
from tensorflow_addons.layers import InstanceNormalization
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, Conv2DTransposePeriodic


def Relu( x, training=None, **kwargs):
    """
    shadow the relu function from tensorflow and allwo for the keyword 'training',
    which otherwise raises a kwarg error
    """
    return relu( x, **kwargs)


def Selu( x, training=None, **kwargs):
    """
    shadow the relu function from tensorflow and allwo for the keyword 'training',
    which otherwise raises a kwarg error
    """
    return selu( x, **kwargs)

class MsPredictor( Model):
    def __init__( self, n_channels, n_out=2, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.upsampler = UpSampling2D()
        self.block = []
        block = self.block
        for i in range( 4):
            block.append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1) ) 
            block.append( InstanceNormalization() ) 
            block.append( Selu )
        block.append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1) ) 
        block.append( InstanceNormalization() ) 
        block.append( Conv2D( n_out, kernel_size=1, strides=1) ) 


    def call( self, image, previous_prediction=None, training=False):
        """
        Predict the layer by giving the machine learned prediction using
        the coarse grained image, and if the prediction of a previous layer
        is given, then use it for 'refinement'
        """
        for layer in self.block:
            image = layer( image, training=training) #already full prediction
        if previous_prediction is not None:
            image = image +  self.upsampler( previous_prediction)
        return image






class UresnetEncoder(Layer):
    def __init__(  self, n_channels, first=False, *args, **kwargs):
        """ 
        Build the residual encoder, if it is the first encoder the leading
        batch normalization has to be droppde
        Parameters:
        -----------
        n_channels:     int,
                        number of channels 
        first:          bool, default False
                        if its the first layer without the leading batch norm
                        and without stride
        """
        super().__init__( *args, **kwargs)
        self.block = []
        block = self.block
        if first:
            stride = 1
        else:
            stride = 2
            block.append( BatchNormalization() )
            block.append( Relu )
        block.append( Conv2DPeriodic( n_channels, kernel_size=3, strides=stride ) )
        block.append( BatchNormalization() )
        block.append( Relu)
        block.append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1) ) 
        self.bypass = Conv2D( n_channels, kernel_size=1, strides=stride)
    
    def __call__ ( self, images, training=False):
        """ evaluate the resnet block """
        x = images
        for layer in self.block:
            x = layer( x, training=training)
        return x + self.bypass( images, training=training)


class UresnetDecoder( Layer):
    def __init__(  self, n_channels, *args, **kwargs):
        """ 
        Build the residual encoder, if it is the first encoder the leading
        batch normalization has to be droppde
        Parameters:
        -----------
        n_channels:     int,
                        number of channels 
        """
        super().__init__( *args, **kwargs)
        self.upsampler = UpSampling2D()
        self.concatenator = Concatenate()
        self.block = []
        block = self.block
        block.append( BatchNormalization() )
        block.append( Relu )
        block.append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1 ) )
        block.append( BatchNormalization() )
        block.append( Relu)
        block.append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1) ) 
        self.bypass = Conv2D( n_channels, kernel_size=1, strides=1)
    
    def __call__ ( self, inputs, bypass, training=False):
        """ evaluate the resnet block """
        inputs = self.upsampler( inputs )
        bypass = self.concatenator( [inputs, bypass] )
        x = bypass
        for layer in self.block:
            x = layer( x, training=training)
        return x + self.bypass( bypass, training=training)
