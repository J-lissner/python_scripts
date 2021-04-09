import tensorflow as tf
import tensorflow_probability as tfp
import itertools
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten
from tensorflow_probability import distributions as tfd



class RegularizedDense(Model):
    def __init__(self, n_output, n_neuron=[16,16], activation=['selu','selu'], dropout=None, batch_normalization=False): #maybe do even "regularizer as input"
        super(RegularizedDense, self).__init__()
        self.architecture = [ Dense( n_neuron[0], activation=activation[0] , kernel_regularizer='l2', kernel_initializer='he_normal')]
        for i in range( 1, len(n_neuron) ): 
            if dropout:
                self.architecture.append( Dropout( dropout) )
            if batch_normalization is True:
                self.architecture.append( BatchNormalization() )

            self.architecture.append( Dense( n_neuron[i], activation=activation[i] , kernel_regularizer='l2', kernel_initializer='he_normal')  )
        self.architecture.append( Dense(n_output, activation=None, kernel_regularizer='l2', kernel_initializer='he_uniform') )


    def call(self, x, training=False):
        for layer in self.architecture:
            x = layer( x, training=training)
        return x

    def predict( self, x):
        return self( x, training=False )


class ForwardNormalization( Model):
    """
    Construct a dense feed forward neural network with (optional) batch normalization and dropout
    The connection to the output layer is always linear #TODO mb can add an optional argument for non linear actuvation function
    """
    def __init__(self, n_output, hidden_neurons, activation, batch_normalization=True, dropout=False):
        """
        Parameters:
        -----------
        n_output:       int
                        number of output neurons of the model
        hidden_neurons: list of ints #TODO bad variable name
                        number of neurons of the internal layers
        activation:     list of strings
                        activation of each layer. Length has to match "hidden_neurons"

        batch_normalization:        bool or list of bools, default False
                                    if a single bool is given, it is applied to every internal layer
                                    if a list of bool is given, length has to match "hidden_neurons" and is specified per layer
        dropout:                    float or list of floats, default False
                                    if a single float is given, it is applied to every internal layer
                                    if a list of float is given, length has to match "hidden_neurons" and is specified per layer 
        """
        super(ForwardNormalization, self).__init__()
        self.architecture = [ Dense( hidden_neurons[0], activation=activation[0] )  ]
        for i in range( 1, len(hidden_neurons) ):
            if isinstance( dropout, float):
                self.architecture.append( Dropout( dropout) )
            elif isinstance( dropout, tuple) or isinstance( dropout, list):
                self.architecture.append( Dropout( dropout[i-1] ) )

            if batch_normalization is True:
                self.architecture.append( BatchNormalization() )
            elif isinstance( batch_normalization, tuple) or isinstance( batch_normalization, list):
                if batch_normalization[i-1] is True:
                    self.architecture.append( BatchNormalization() )

            self.architecture.append( Dense( hidden_neurons[i], activation=activation[i] )  ) 
        self.architecture.append( Dense(n_output, activation=None) )


    def call(self, x, training=False):
        for layer in self.architecture:
            x = layer( x, training=training)
        return x

    def predict( self, x):
        return self( x, training=False )
        


class ReconstructedANN( Model):
    def __init__(self, n_output, n_neuron=[6,7], activation=['softplus','softplus'], *args, **kwargs): 
        super( ReconstructedANN, self).__init__()
        self.architecture = []
        for i in range( len( n_neuron) ):
            self.architecture.append( Dense( n_neuron[i], activation=activation[i] ) )
        self.architecture.append( Dense( n_output, activation=None ) )

    def call(self, x, training=False):
        for layer in self.architecture:
            x = layer(x)
        return x

    def predict( self, x):
        return self.call( x, training=False)


############ Bayesian Neural Networks (BNN) ##############
class BayesianNN( Model):
    """
    Constructs a Bayesian Neural Network
    The penultimate layer has 'tanh' activation (for small NNs, linear and 
    other non-linear activations did not result in sensible training)
    The output of this model is of type tensorflow_probability.distribution. 
    (Look into tfp docs for more methods and attributes)
    """
    def __init__(self, n_output,  KLD_func, n_neuron=[6,7], activation=['selu'], batch_norm=False, *args, **kwargs): 
        """
        Parameters:
        ---------------------
        n_output:       int
                        number of output neurons of the model 
        KLD_func:       lambda function
                        Kullback-Liebler Divergence function of Tensorflow scaled by the training set size 
        n_neurons:      list of ints
                        no. of neurons in hidden layer (excl. input and output layers) 
        activation:     list of strings
                        activation of each layer. Endlessly cycles the 
                        activation functions if len( activation) < len(n_neuron)
        batch_norm:     bool, default False
                        if batch normalization should be applied behind each hidden layer
        """
        super( BayesianNN, self).__init__()
        self.architecture = []
        activation = itertools.cycle( activation)
        for i in range( len( n_neuron) ):
            self.architecture.append( tfp.layers.DenseFlipout( n_neuron[i], activation=next(activation), dtype='float64', 
                                                         kernel_divergence_fn=KLD_func, bias_divergence_fn=KLD_func,
                                                         bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn) )
            if batch_norm: 
                self.architecture.append( BatchNormalization() )
        self.architecture.append( tfp.layers.DenseFlipout( 2*n_output, activation=next(activation), dtype='float64',
                                                     kernel_divergence_fn=KLD_func, bias_divergence_fn=KLD_func,
                                                     bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                                                     bias_prior_fn=tfp.layers.default_multivariate_normal_fn) )
        self.architecture.append( tfp.layers.DistributionLambda( make_distribution_fn=lambda params: tfd.Normal(loc=params[...,:n_output],
                                                           scale= 1e-3 + tf.abs(params[...,n_output:])), dtype='float64' )) 

    def call(self, x, training=False):
        for layer in self.architecture:
            x = layer(x)
        return x

    def predict_validation( self, x):
        return self.call( x, training=False)


class ProbabalisticNN( Model):
    """
    Constructs a Dense feedforward neural network which predicts a distribution
    The output of this model is of type tensorflow_probability.distribution. 
    (Look into tfp docs for more methods and attributes)
    """
    def __init__(self, n_output, n_neuron=[6,7], activation=['selu'], batch_norm=False, *args, **kwargs): 
        """
        Parameters:
        ---------------------
        n_output:       int
                        number of output neurons of the model 
        n_neurons:      list of ints
                        no. of neurons in hidden layer (excl. input and output layers) 
        activation:     list of strings
                        activation of each layer. Endlessly cycles the 
                        activation functions if len( activation) < len(n_neuron)
        batch_norm:     bool, default False
                        if batch normalization should be applied behind each hidden layer
        """
        super( ProbabalisticNN, self).__init__()
        self.architecture = []
        activation = itertools.cycle( activation)
        distribution = lambda params: tfd.Normal(loc=params[...,:n_output],
                                 scale= 1e-3 + tf.abs(params[...,n_output:]))
                                 
        for i in range( len( n_neuron) ):
            self.architecture.append( Dense( n_neuron[i], activation=next(activation), dtype='float64' ) )
            if batch_norm: 
                self.architecture.append( BatchNormalization() )
        self.architecture.append( Dense( 2*n_output, activation=next(activation), dtype='float64' ) )
        self.architecture.append( tfp.layers.DistributionLambda( make_distribution_fn=tfp.layers.DistributionLambda( distribution, dtype='float64')  ) )

    def call(self, x, training=False):
        for layer in self.architecture:
            x = layer(x)
        return x

    def predict_validation( self, x):
        return self.call( x, training=False)

class AlexNet(Model):
    def __init__( self, output_size ):
        """
        set up the alex net (which might not be 100% alex net but really close to it)
        """
        super(AlexNet, self).__init__()
        arch = [] #abbreviation for self.architecture
        ## feature extraction
        arch.append( Conv2D( 64, kernel_size=(11,11), strides=(4,4), padding='same', activation='relu' ))
        arch.append( MaxPool2D( pool_size=(3,3), strides=(2,2) ))
        arch.append( Conv2D( 192, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu' ))
        arch.append( MaxPool2D( pool_size=(3,3), strides=(2,2) ))
        arch.append( Conv2D( 384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu' ))
        arch.append( Conv2D( 256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu' )) 
        arch.append( MaxPool2D( pool_size=(3,3), strides=(2,2) ) )
        ## pooling
        arch.append( AveragePooling2D(3,3) ) 
        ## regressor
        arch.append( Flatten() )
        arch.append( Dropout( 0.5) )
        arch.append( Dense( 4096, activation='relu') )
        arch.append( Dropout( 0.5) )
        arch.append( Dense( 4096, activation='relu') )
        arch.append( Dense( output_size) )
        self.architecture = arch
        

    def call(self, x, training=False):
        for layer in self.architecture:
            x = layer( x, training=training)
        return x

    def predict( self, x):
        return self( x, training=False )


############ BELOW HERE YOU WILL FIND ONLY TRASH WHICH WAS HERE FOR TESTING

class DummyNN( Model):
    #def __init__(self, n_output, n_neuron=[16,16], activation=['selu','selu'], dropout=None): #maybe do even "regularizer as input"
    #    super(DummyNN, self).__init__()
    #    self.architecture = []
    #    for i in range( len(n_neuron) ):
    #        self.architecture.append( Dense( n_neuron[i], activation=activation[i] , kernel_regularizer='l2', kernel_initializer='he_normal')  )
    #        if dropout:
    #            self.architecture.append( Dropout( dropout) )
    #    self.architecture.append( Dense(n_output, activation=None, kernel_regularizer='l2', kernel_initializer='he_uniform') )
    #    if dropout:
    #        self.architecture.append( Dropout( dropout) )

    def __init__( self, n_output=2, *args, **kwargs):
        super(DummyNN, self).__init__() 
        self.architecture = []
        self.architecture.append( Dense( 12, activation='selu') )
        self.architecture.append( Dropout( 0.5) )
        self.architecture.append( Dense( 8, activation='selu') )
        self.architecture.append( Dense( n_output) )
    #def __init__( self, n_output=2):
    #    super(DummyNN, self).__init__() 
    #    self.d1 = Dense( 12, activation='selu')
    #    self.d2 = Dense( 8, activation='selu')
    #    self.d3 = Dense( n_output)

    def call(self, x, training=False):
        if training is True:
            for layer in self.architecture:
                x = layer(x)
        elif training is False:
            for layer in self.architecture:
                if isinstance( layer, Dense):
                    x = layer( x)
        return x

    #def call( self, x, *args, **kwargs):
    #    for layer in self.architecture:
    #        x = layer( x)
    #    return x
    #def call( self, x, *args, **kwargs ):
    #    x = self.d1( x)
    #    x = self.d2( x)
    #    x = self.d3( x)
    #    return x

    @tf.function
    def train_myself( self, x, y, cost, optimizer):
        with tf.GradientTape() as tape:
            prediction = self.call( x)
            loss = cost( y, prediction)
        gradients = tape.gradient( loss, self.trainable_variables)
        optimizer.apply_gradients( zip( gradients, self.trainable_variables ) )
        return loss

    def predict( self, x):
        return self.call( x)

