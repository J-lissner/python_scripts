import tensorflow as tf
import itertools
from tensorflow.math import ceil
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, Flatten, Concatenate
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic
from hybrid_models import VolBypass


# overwrite the model to get the base call and predict_validation for default behavious
class Model( Model):
    """
    This is used to implement the generic methods like
    call, predict, predict_validation
    and inherit them, that I have less copied lines of code
    """
    def __init__( self, *args, **kwargs):
        super( Model, self).__init__()
        self.architecture = []

    def call(self, x, training=False, *args, **kwargs):
        for layer in self.architecture:
            x = layer( x, training=training)
        return x

    def predict( self, x):
        return self( x, training=False )

    def predict_validation( self, x):
        return self( x, training=False )



class RegularizedDense(Model):
    def __init__(self, n_output, n_neuron=[16,16], activation=['selu','selu'], dropout=None, batch_norm=False): #maybe do even "regularizer as input"
        super(RegularizedDense, self).__init__()
        self.architecture = [ Dense( n_neuron[0], activation=activation[0] , kernel_regularizer='l2', kernel_initializer='he_normal')]
        model = self.architecture
        for i in range( 1, len(n_neuron) ): 
            if dropout:
                model.append( Dropout( dropout) )
            if batch_norm is True:
                model.append( BatchNormalization() )

            model.append( Dense( n_neuron[i], activation=activation[i] , kernel_regularizer='l2', kernel_initializer='he_normal')  )
        model.append( Dense(n_output, activation=None, kernel_regularizer='l2', kernel_initializer='he_uniform') )



class ForwardNormalization( Model):
    """
    Construct a dense feed forward neural network with (optional) batch normalization and dropout
    The connection to the output layer is always linear #TODO mb can add an optional argument for non linear actuvation function
    """
    def __init__(self, n_output, hidden_neurons, activation, batch_norm=True, dropout=False):
        """
        Parameters:
        -----------
        n_output:       int
                        number of output neurons of the model
        hidden_neurons: list of ints #TODO bad variable name
                        number of neurons of the internal layers
        activation:     list of strings
                        activation of each layer. Length has to match "hidden_neurons" 
        batch_norm:    bool or list of bools, default False
                       if a single bool is given, it is applied to every internal layer
                       if a list of bool is given, length has to match "hidden_neurons"
                       and is specified per layer
        dropout:       float or list of floats, default False
                       if a single float is given, it is applied to every internal layer
                       if a list of float is given, length has to match "hidden_neurons" 
                       and is specified per layer 
        """
        super(ForwardNormalization, self).__init__()
        activation = itertools.cycle( activation)
        self.architecture = [ Dense( hidden_neurons[0], activation=next(activation) )  ]
        model = self.architecture
        for i in range( 1, len(hidden_neurons) ):
            if isinstance( dropout, float):
                model.append( Dropout( dropout) )
            elif isinstance( dropout, tuple) or isinstance( dropout, list):
                model.append( Dropout( dropout[i-1] ) )

            if batch_norm is True:
                model.append( BatchNormalization() )
            elif isinstance( batch_norm, tuple) or isinstance( batch_norm, list):
                if batch_norm[i-1] is True:
                    model.append( BatchNormalization() )

            model.append( Dense( hidden_neurons[i], activation=next(activation) )  ) 
        model.append( Dense(n_output, activation=None) )

        
class ReconstructedANN( Model):
    """
    Model which served as a baseline which was solely reconstructed from the 
    paper data driven microstructure property relations (2018)
    """
    def __init__(self, n_output, n_neuron=[6,7], activation=['softplus','softplus'], *args, **kwargs): 
        super( ReconstructedANN, self).__init__()
        self.architecture = []
        for i in range( len( n_neuron) ):
            self.architecture.append( Dense( n_neuron[i], activation=activation[i] ) )
        self.architecture.append( Dense( n_output, activation=None ) )



##################### Autoencoders ##################### 
class AutoEncoder( Model):
    def __init__( self, n_encode, input_dim, encoded_dim, dropout=0.5, activation='selu'):
        """
        get the architecture of a autoencoder
        mirrors the encoding architecture to the decoding architecture as of now
        Parameters:
        -----------
        n_encode:       int
                        number of layers to encode
        input_dim:      int
                        dimension of the input data
        encoded_dim:    int
                        size of the encode layer
        dropout:        float or bool, default 0.5
                        if and what dropout should be applied
        activation:     string or list of strings or bool, default 'selu'
                        which activation to apply. If only a string is 
                        given then all layers share the same activation
                        function. Otherwise a list specifies each layer
        
        """
        super( AutoEncoder, self).__init__()
        #input preprocessing
        layer_decline = (input_dim - encoded_dim)/(n_encode+1)
        if isinstance( activation, str):
            activation = (2*n_encode+1)*[activation]
        self.architecture = []
        model = self.architecture
        #encode
        for i in range( n_encode ):
            model.append( Dense( ceil( input_dim - (i+1)*layer_decline), activation=activation[i] ) )
            if dropout:
                model.append( Dropout( dropout))
        #middle layer
        model.append( Dense( encoded_dim, activation=activation[i+1] ) )
        #decode
        for i in range( n_encode):
            model.append( Dense( ceil(encoded_dim + (i+1)* layer_decline), activation=activation[i] ) )
            if dropout:
                model.append( Dropout( dropout) )
        #output layer
        model.append( Dense( input_dim) )



##################### Convolutional Neural Networks ##################### 
class InceptionLike( Model):
  def __init__( self, n_output, input_size, activation=None, n_blocks=1, globalpool=False, *args, **kwargs):
    """
    n_output: int, size output
    input_size: int, size input
    activation: activation function in all layers
    globalpool: if global average pooling should be conducted before flattening 
    """
    super( InceptionLike, self).__init__()
    self.n_blocks = n_blocks
    self.activation = activation
    self.input_size = input_size
    self.n_output = n_output
    self.globalpool = globalpool
    self.build_model()


  def build_model( self):
    #### First inception block
    self.inception_1 = [ [], [], [], [] ]
    block_layer = self.inception_1[0]
    block_layer.append( Conv2DPeriodic( filters=5, kernel_size=17, strides=10, activation=self.activation, input_shape=self.input_size)) 
    block_layer.append( MaxPool2DPeriodic( pool_size=2, strides=None) ) #None defaults to pool_size) 
    block_layer.append( BatchNormalization() )
    block_layer.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=2, activation=self.activation))
    block_layer.append( BatchNormalization() )

    #second middle filter generic filter pooling filter
    generic = self.inception_1[1]
    generic.append( Conv2DPeriodic( filters=5, kernel_size=5, strides=3, activation=self.activation, input_shape=self.input_size))
    generic.append( MaxPool2DPeriodic( pool_size=2))
    generic.append( BatchNormalization() )
    generic.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=3, activation=self.activation))
    generic.append( MaxPool2DPeriodic( pool_size=2, padding='valid') )
    generic.append( BatchNormalization() )

    # third block with average pooling and a medium sized filter
    avg_medium = self.inception_1[2]
    avg_medium.append( AvgPool2DPeriodic( pool_size=5, input_shape = self.input_size))
    avg_medium.append( Conv2DPeriodic( filters=5, kernel_size=5, strides=2, activation=self.activation ) )
    avg_medium.append( BatchNormalization() )
    avg_medium.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=2, activation=self.activation ) )
    avg_medium.append( MaxPool2DPeriodic( pool_size=2) )
    avg_medium.append( BatchNormalization() )

    # third block with average pooling and a medium sized filter
    avg_small = self.inception_1[3]
    avg_small.append( AvgPool2DPeriodic( pool_size=5, input_shape=self.input_size))
    avg_small.append( BatchNormalization() )
    avg_small.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=2, activation=self.activation ) )
    avg_small.append( AvgPool2DPeriodic( pool_size=4) )
    avg_small.append( BatchNormalization() )
    ########

    self.dense = []
    if self.globalpool is True:
        self.dense.append( GlobalAveragePooling2D() )
    self.dense.append( Flatten() )
    self.dense.append( Dense( 128, activation='selu') )
    self.dense.append( BatchNormalization() )
    self.dense.append( Dense( 64, activation='selu') )
    self.dense.append( BatchNormalization() )
    self.dense.append( Dense( 32, activation='selu') )
    self.dense.append( BatchNormalization() ) 
    self.dense.append( Dense( 16, activation='selu') ) 
    self.dense.append( BatchNormalization() ) 
    self.dense.append( Dense( self.n_output) ) 


  def call( self, images):
    #convolutions of feature extractions
    x = []
    for i in range( len( self.inception_1)):
        for j in range( len( self.inception_1[i]) ):
            if j == 0:
                x.append( self.inception_1[i][j]( images) )
            else:
                x[i] = self.inception_1[i][j]( x[i] ) 

    # concatenation after first inception layer
    x = concatenate( x, axis=-1 )#
    if self.n_blocks == 2:
        pass 

    #dense layer prediction
    #x = concatenate( x), vol, k_1, k_2)
    for layer in self.dense:
        x = layer(x)
    return x



class TranslationInvariant( VolBypass):
  """ 
  here we attempt to have a fully translational invariant neural network
  by implementing periodic padding etc. and using globalpool at the end to
  lose the relative position of each convolution feature. 
  It turns out that it does not quite work that way and we only achieve 
  true a priori transltion invariance for no stride and globalpool (which
  is a shit prescription on the model because then the resolution doesn't 
  downsample and we have so many fucking pixels throughout the whole image
  """
  def __init__( self, n_output, n_vol=1, *args, **kwargs):
    super().__init__( n_output, n_vol, *args, **kwargs)
    del self.regressor
    self.build_cnn()
    
  def build_cnn( self):
    """ build two blocks of inception modules with an average globalpooling at the end"""
    stride = 1  #for true translational invariance 
    self.inception1 = [ [], [], [], [], []]
    huge = self.inception1[0]
    huge.append( Conv2DPeriodic( filters=10, kernel_size=15, strides=stride, activation='relu') )
    huge.append( Conv2DPeriodic( filters=10, kernel_size=11, strides=stride, activation='relu') )
    huge.append( Conv2DPeriodic( filters=10, kernel_size=7, strides=stride, activation='relu') )
    huge.append( Conv2DPeriodic( filters=10, kernel_size=5, strides=stride, activation='relu') )
    large = self.inception1[1]
    large.append( Conv2DPeriodic( filters=10, kernel_size=11, strides=stride, activation='relu') )
    large.append( Conv2DPeriodic( filters=10, kernel_size=7, strides=stride, activation='relu') )
    large.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=stride, activation='relu') )
    medium= self.inception1[2]
    medium.append( Conv2DPeriodic( filters=10, kernel_size=7, strides=stride, activation='relu') )
    medium.append( Conv2DPeriodic( filters=10, kernel_size=5, strides=stride, activation='relu') )
    medium.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=stride, activation='relu') )
    small = self.inception1[3]
    small.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=stride, activation='relu') )
    small.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=stride, activation='relu') )
    layer_concat = self.inception1[-1]
    layer_concat.append( Concatenate())
    layer_concat.append( Conv2D( filters=30, kernel_size=1, strides=stride, activation='selu'))
    layer_concat.append( BatchNormalization() )

    self.inception2 = [ [], [], [], [] ]
    big = self.inception2[0]
    big.append( Conv2DPeriodic( filters=10, kernel_size=7, strides=stride, activation='relu') )
    big.append( Conv2DPeriodic( filters=10, kernel_size=5, strides=stride, activation='relu') )
    medium = self.inception2[1]
    medium.append( Conv2DPeriodic( filters=10, kernel_size=5, strides=stride, activation='relu') )
    medium.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=stride, activation='relu') )
    small = self.inception2[2]
    small.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=stride, activation='relu') )
    small.append( Conv2DPeriodic( filters=10, kernel_size=3, strides=stride, activation='relu') )
    layer_concat = self.inception2[-1]
    layer_concat.append( Concatenate())
    layer_concat.append( Conv2D( filters=15, kernel_size=1, strides=stride, activation='selu'))
    layer_concat.append( BatchNormalization() )
    layer_concat.append( Flatten() )

    self.regressor = []
    self.regressor.append( Dense( 1000, activation='selu' )) 
    self.regressor.append( BatchNormalization()) 
    self.regressor.append( Dense( 500, activation='selu' )) 
    self.regressor.append( BatchNormalization()) 
    self.regressor.append( Dense( 200, activation='selu' )) 
    self.regressor.append( BatchNormalization()) 
    self.regressor.append( Dense( 100, activation='selu' )) 
    self.regressor.append( BatchNormalization()) 
    self.regressor.append( Dense( 50, activation='selu' )) 
    self.regressor.append( BatchNormalization()) 
    self.regressor.append( Dense( self.n_output ) )


  def call( self, vol, images, training=False, *args, **kwargs):
    x_vol = self.predict_vol( vol, training=training)
    x_cnn = self.predict_cnn( images, training=training)
    return x_vol + x_cnn

  def predict_cnn( self, images, training=False):
    x_1 = []
    for i in range( len( self.inception1) -1 ):
        for j in range( len( self.inception1[i]) ):
            if j == 0:
                x_1.append( self.inception1[i][j]( images, training=training) )
            else:
                x_1[i] = self.inception1[i][j]( x_1[i], training=training ) 
    #concatenation and 1x1 convo
    for layer in self.inception1[-1]:
        x_1 = layer( x_1, training=training)
    x = []
    for i in range( len( self.inception2) -1 ):
        for j in range( len( self.inception2[i]) ):
            if j == 0:
                x.append( self.inception2[i][j]( x_1, training=training) )
            else:
                x[i] = self.inception2[i][j]( x[i], training=training ) 
    del x_1
    #concatenation and 1x1 convo
    for layer in self.inception2[-1]:
        x = layer( x, training=training)
    # regression of fatures
    for layer in self.regressor:
        x = layer(x, training=training)
    return x



class GenericCnn( Model):
  def __init__( self, n_output, dense=[256,128,64,32], activation='selu', batch_norm=True, pre_pool=0, **conv_architecture ):
    """
    Get a generic deep conv net
    Parameters:
    -----------
    n_output:       int
                    size of output
    dense:          list like of ints, default [256,128,64,32]
                    dense layer after pooling
    activation:     str or list like of str, default 'selu'
                    activation function for dense layer 
    batch_norm:     bool, default True
                    whether to apply batch normalization after dense and pooling
    pre_pool:       int, default 0
                    downsampling of resolution of input image via average pooling
    **conv_architecture with default values:
    kernels:        list like of ints, default [11,7,5,3,3]
                    dimension of each kernel, len(kernels) == n_conv
    strides:        list like of ints, default [4,3,3,2,2]
                    stride of each kernel, has to match len( kernels)
    filters:        list like of ints, default [32,32,64,64,96]
                    number of channels per layer, has to match len( kernels)
    pooling:        bool or list of ints, default True
                    if booling should be applied after every layer,
                    can be specified with ints
    Returns:
    --------
    None:           builds the model in self.architecture
    """
    #then also try if a parallel model with downsampled inputs is good
    super().__init__()
    model = self.architecture #gotten from super
    ## input preprocessing
    kernels = conv_architecture.pop( 'kernels', [11,7,5,3,3])
    strides = conv_architecture.pop( 'strides', [4,3,3,2,2])
    filters = conv_architecture.pop( 'filters', [32,32,64,64,96])
    pooling = conv_architecture.pop( 'pooling', True)
    n_conv = len( kernels)
    n_dense = len( dense)
    if isinstance( pooling, bool) and pooling is True:
        pooling = n_conv*[2]
    elif isinstance( pooling, int):
        pooling = n_conv*[pooling]
    else:
        pooling = n_conv*[False]
    if isinstance( activation, str) or activation is None:
        activation = (n_conv+n_dense)*[activation]
    ## model building
    if pre_pool:
        model.append( AvgPool2DPeriodic( pre_pool) )
    for i in range( n_conv):
        model.append( Conv2DPeriodic( filters[i], kernels[i], strides[i], activation=activation[i] ) )
        if pooling[i]:
            model.append( MaxPool2DPeriodic( pooling[i] ) )
    model.append( Flatten() )
    model.append( BatchNormalization())
    for i in range( n_dense):
        model.append( Dense( dense[i], activation=activation[i+n_conv]) )
        if batch_norm:
            model.append(  BatchNormalization() )
    model.append( Dense( n_output) )



