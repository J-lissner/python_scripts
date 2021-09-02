import tensorflow as tf
import itertools
from tensorflow.math import ceil
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate, Flatten
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic


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

    def call(self, x, training=False):
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
  def __init__( self, output_size, input_size, activation=None, n_blocks=1, globalpool=False, *args, **kwargs):
    """
    output_size: int, size output
    input_size: int, size input
    activation: activation function in all layers
    globalpool: if global average pooling should be conducted before flattening 
    """
    super( InceptionLike, self).__init__()
    self.n_blocks = n_blocks
    self.activation = activation
    self.input_size = input_size
    self.output_size = output_size
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
    self.dense.append( Dense( self.output_size) ) 


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



class TranslationInvariant( Model):
  def __init__( self, output_size, input_size, strides=1, globalpool=True, downsample=False, *args, **kwargs):
    super().__init__()
    
    self.architecture = []
    model = self.architecture
    if downsample:
        model.append( AvgPool2DPeriodic( downsample) )
    model.append( Conv2DPeriodic( filters=5, kernel_size=7, strides=strides, input_shape=input_size) )
    model.append( Conv2DPeriodic( filters=10, kernel_size=5, strides=strides) )
    model.append( Conv2DPeriodic( filters=15, kernel_size=3, strides=strides) )
    #model.append( Conv2DPeriodic( filters=20, kernel_size=1, strides=strides) )
    if globalpool:
        model.append( GlobalAveragePooling2D() )
    model.append( Flatten() )
    #model.append( Dense( 128) )
    #model.append( Dense( 64) )
    model.append( Dense( 32, activation='selu') )
    model.append( Dense( 16, activation='selu') )
    model.append( Dense( output_size) ) 

  def call( self, x):
      for layer in self.architecture:
          x = layer( x)
      return x


