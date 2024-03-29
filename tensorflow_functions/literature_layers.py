import tensorflow as tf
from tensorflow.keras.activations import relu, selu
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, Concatenate, Add, Average
from tensorflow_addons.layers import InstanceNormalization
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, Conv2DTransposePeriodic
from my_layers import SqueezeExcite, LayerWrapper


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

class InceptionModule( Layer):
    """ 
    Literature inception module with 1x1 bypass, 3x3/5x5 conv with preceeding 
    1x1 reduction, and 3x3 maxpool with postceeding 1x1 conv
    """
    def __init__( self, n_channels, *args, **kwargs):
        """
        Invoke the architecture of the module. The n_channels parameter 
        is highly flexible and behaves different based on lengths
        Parameters:
        -----------
        n_channels:     int, or list of ints
                        how many channels per branch. 
                        If an int is given, all layers have this many channels
                        If a list of 2 is given, it defines the reduction (on 3x3/5x5) and 
                        branch output of each branch in the second index
                        if a list of [int, list, list, int] is given each nest corresponds to 
                        the branch in this order: 1x1bypass, 3x3, 5x5, pool-1x1 
        """
        super().__init__( *args, **kwargs)
        if isinstance( n_channels, int):
            n_channels = [ n_channels, 2*[n_channels], 2*[n_channels], n_channels ]
        elif len( n_channels) == 2:
            n_channels = [ n_channels[1], n_channels, n_channels, n_channels[1] ]
        Conv1x1 = lambda n: Conv2D( n, kernel_size=1)
        self.architecture = LayerWrapper( [Conv1x1( n_channels[0] )] )
        self.architecture[-1].append( [Conv1x1( n_channels[1][0]), Conv2DPeriodic( n_channels[1][1], kernel_size=3)] )
        self.architecture[-1].append( [Conv1x1( n_channels[2][0]), Conv2DPeriodic( n_channels[2][1], kernel_size=5)] )
        self.architecture[-1].append( [MaxPool2DPeriodic( 3, strides=1), Conv1x1( n_channels[-1]) ]  )
        self.architecture.append( Concatenate() )

    def __call__( self, images, **layer_kwargs):
        """
        Evaluate the inception module (oneliner through LayerWrapper)
        """
        return self.architecture( images, **layer_kwargs) 

class MsPredictor( Layer):
    def __init__( self, n_channels, n_out=2, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.upsampler = UpSampling2D()
        self.concatenator = Concatenate()
        self.block = []
        for i in range( 4):
            self.block.append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1) ) 
            self.block.append( InstanceNormalization() ) 
            self.block.append( Selu )
        self.block.append( Conv2DPeriodic( n_channels, kernel_size=3, strides=1) ) 
        self.block.append( InstanceNormalization() ) 
        self.block.append( Conv2D( n_out, kernel_size=1, strides=1) ) 


    def __call__( self, image, previous_prediction=None, training=False):
        """
        Predict the layer by giving the machine learned prediction using
        the coarse grained image, and if the prediction of a previous layer
        is given, then use it for 'refinement'
        """
        if previous_prediction is not None:
            previous_prediction = self.upsampler( previous_prediction)
            image = self.concatenator( [image, previous_prediction] )
        for layer in self.block:
            image = layer( image, training=training) #already full prediction
        if previous_prediction is not None:
            image = image + previous_prediction
        return image


    def freeze( self, freeze=True):
        for layer in self.block:
            try: 
                layer.trainable = not freeze
            except: 
                print( 'unable to freeze', layer) #its an activation function




class ResBlock( Layer):
    """
    Block of the deep residual network directly taken from the literature
    CARE: this is the implementation of the bottleneck layer,
    (with one less 3x3 conv but 1x1 pre-post conv for dimensionality reduction)
    not the residual block with 2 3x3 conv
    """
    def __init__( self, n_out, strides=1, sne=False, blowup=False, concatenator=Add, *args, **kwargs):
        """
        Parameters:
        -----------
        n_out:          int,
                        number of output channels, divides by 4 for intermediate operations
        strides:        int, default 1
                        if stride> 1, then the bypass is done via 1x1 conv (no activation function)
        sne:            bool, default False
                        whether the squeeze adn excitation block should be added after convolution
        blowup:         bool, default False
                        whether to expand the number of channels via 1x1 conovolution (required
                        for first res block)
        """
        super().__init__( *args, **kwargs)
        n_channels = n_out // 4
        self.concatenator = concatenator()
        self.block = LayerWrapper()
        self.block.append( Conv2D( n_channels, kernel_size=1, strides=strides ) )
        self.block.append( BatchNormalization( ) )
        self.block.append( tf.keras.layers.Activation( 'relu' ) )
        self.block.append( Conv2DPeriodic( n_channels, kernel_size=3) )
        self.block.append( BatchNormalization( ) )
        self.block.append( tf.keras.layers.Activation( 'relu' ) )
        self.block.append( Conv2D( n_out, kernel_size=1) )
        self.block.append( BatchNormalization( ) )
        if sne:
            self.block.append( SqueezeExcite( n_out) )
        ### bypass etc.
        if strides > 1 or blowup:
            self.bypass = Conv2D( n_out, kernel_size=1, strides=strides )
        else: # identity mapping
            self.bypass = lambda x, *args, **kwargs: x
        self.relu = tf.keras.layers.Activation( 'relu' ) 


    def __call__( self, images, *args, training=False, **kwargs):
        images = self.concatenator( [self.block( images, training=training), self.bypass( images, training=training)] )
        return self.relu( images)


class ResxBlock( ResBlock):
    """
    implements the bottleneck resXblock of the resXnet. has more channels
    and more 3x3 operations than the resblock.
    """
    def __init__( self, n_out, strides=1, cardinality=32, sne=False, blowup=False, *args, **kwargs):
        """
        see above, inherited from ResBlock, pracitcally the same
        """
        super().__init__( n_out, strides=strides, sne=sne, blowup=blowup, *args, **kwargs)
        self.block = LayerWrapper( [] )
        n_channels = n_out // 2 
        n_branch = n_channels // cardinality
        for i in range( cardinality):
            resx_block = []
            resx_block.append( Conv2D( n_branch, kernel_size=1, strides=strides) )
            resx_block.append( BatchNormalization() )
            resx_block.append( tf.keras.layers.Activation( 'relu' ) )
            resx_block.append( Conv2DPeriodic( n_branch, kernel_size=3) )
            resx_block.append( BatchNormalization() )
            self.block[-1].append( resx_block )
        self.block.append( Concatenate() ) 
        self.block.append( Conv2D( n_out, kernel_size=1 )  )
        self.block.append( BatchNormalization() ) 
        if sne:
            self.block.append( SqueezeExcite( n_out) )
        #bypass n relu inherited


        



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
