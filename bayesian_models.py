import tensorflow as tf
import tensorflow_probability as tfp
import itertools
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow_probability import distributions as tfd
from tensorflow.math import ceil


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
        model = self.architecture
        distribution = lambda x: tfd.Normal(loc=x[...,:n_output], scale= 1e-3 + tf.abs(x[...,n_output:])) 
                                 #scale= 1e-3 + tf.abs(params[...,n_output:])) 
        activation = itertools.cycle( activation)
        for i in range( len( n_neuron) ):
            model.append( tfp.layers.DenseFlipout( n_neuron[i], activation=next(activation) ) )
            #model.append( tfp.layers.DenseFlipout( n_neuron[i], activation=next(activation), dtype='float64', 
            #                                             kernel_divergence_fn=KLD_func, bias_divergence_fn=KLD_func,
            #                                             bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
            #                                             bias_prior_fn=tfp.layers.default_multivariate_normal_fn) )
            if batch_norm: 
                model.append( BatchNormalization() )
        model.append( tfp.layers.DenseFlipout( n_output, activation=next(activation) ) )
        ##model.append( tfp.layers.DenseFlipout( 2*n_output, activation=next(activation), dtype='float64',
        ##                             kernel_divergence_fn=KLD_func, bias_divergence_fn=KLD_func,
        ##                             bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
        ##                             bias_prior_fn=tfp.layers.default_multivariate_normal_fn) )
        #model.append( tfp.layers.DenseFlipout( 2*n_output, activation=next(activation) ) )
        #model.append( tfp.layers.DistributionLambda( make_distribution_fn=tfp.layers.DistributionLambda( distribution, dtype='float64')  ) )

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
        -----------
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
        model = self.architecture
        activation = itertools.cycle( activation)
        distribution = lambda params: tfd.Normal(loc=params[...,:n_output],
                                 scale= 1e-3 + tf.abs(params[...,n_output:])) 
        for i in range( len( n_neuron) ):
            model.append( Dense( n_neuron[i], activation=next(activation), dtype='float64' ) )
            if batch_norm: 
                model.append( BatchNormalization() )
        model.append( Dense( 2*n_output, activation=next(activation), dtype='float64' ) )
        model.append( tfp.layers.DistributionLambda( make_distribution_fn=tfp.layers.DistributionLambda( distribution, dtype='float64')  ) )


    def call(self, x, training=False):
        for layer in self.architecture:
            x = layer(x)
        return x

    def predict_validation( self, x):
        return self.call( x, training=False)

