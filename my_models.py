import tensorflow as tf
import numpy as np
from tensorflow.math import ceil
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, concatenate
from other_functions import Cycler
import data_processing as get


# overwrite the model to get default behaviour
class Model( Model):
  """
  This is used to implement the generic methods like call, batched_prediction, etc.
  This class should be only inherited and can not be used as standalone
  """
  def __init__( self, n_output, *args, **kwargs):
      """
      Parameters:
      -----------
      n_output:     int
                    how many values to predict
      """
      super( Model, self).__init__()
      self.n_output = n_output
      self.architecture = []


  def batched_prediction( self, batchsize, *inputs, predictor=None, **kwargs):
      """
      predict the given data in batches and return the prediction
      takes variable inputs because this method is inherited to more
      complicated models.
      Parameters:
      -----------
      batchsize:  int
                  how large the batches should be
      *inputs:    list of tf.tensor like
                  input data to predict
      **kwargs:   other keyworded options for the call,
                  also takes input data
      Returns:
      --------
      prediction: tensorflow.tensor
                  prediction of the model when using self.call()
      """
      if predictor is None:
          predictor = self
      n_batches =  int(inputs[0].shape[0]// batchsize)
      if n_batches == 1:
          return predictor( *inputs, **kwargs)
      prediction = []
      n_samples = inputs[0].shape[0] if inputs else kwargs.items()[0].shape[0]
      jj = 0 #to catch 1 batch
      for i in range( n_batches-1):
          ii = i* n_samples//n_batches
          jj = (i+1)* n_samples//n_batches
          sliced_args = get.slice_args( ii, jj, *inputs)
          sliced_kwargs = get.slice_kwargs( ii, jj, **kwargs) 
          prediction.append( predictor( *sliced_args, **sliced_kwargs ) )
      sliced_args = get.slice_args( jj, None, *inputs)
      sliced_kwargs = get.slice_kwargs( jj, None, **kwargs) 
      prediction.append( predictor( *sliced_args, **sliced_kwargs ) )
      return concatenate( prediction, axis=0) 


  def freeze( self, freeze=True):
      """ freeze the generic model which is given in any default ann"""
      for layer in self.architecture:
          layer.trainable = not freeze

  def freeze_all( self, freeze=True):
      """ 
      freeze or unfreeze the whole model by calling upon every method
      that contains the word 'freeze'
      """
      freeze_methods = [method for method in dir( self) if 'freeze' in method.lower()]
      freeze_methods.pop( freeze_methods.index( 'freeze_all') )
      for method in freeze_methods:
          method = getattr( self, method)
          method( freeze)

  ## call and shadows of call
  def call(self, x, training=False, *args, **kwargs):
      if training is False and False:#TODO: implement the scaling when not training, seems real nice
          x = get.scale_with_shifts( x, self.input_scaling )
          #and then below
    
      for layer in self.architecture:
          x = layer( x, training=training)
      if training is False and False:
          x = get.unscale_data( x, self.output_scaling) 
      return x

  def predict( self, x):
      return self( x, training=False )

  def predict_validation( self, x):
      return self( x, training=False )




class RegularizedDense(Model):
    def __init__(self, n_output, n_neuron=[45,32,25], activation='selu', dropout=None, batch_norm=True, **kwargs): #maybe do even "regularizer as input"
        """
        Whatever dense model with batch normalization in the middle
        """
        super(RegularizedDense, self).__init__(n_output)
        activation = [activation] if not isinstance( activation,list) else activation
        dropout = 0.5 if dropout is True else dropout
        activation = Cycler( activation)
        self.architecture = [ Dense( n_neuron[0], activation=next(activation))]
        model = self.architecture
        for i in range( 1, len(n_neuron) ):
            if dropout:
                model.append( Dropout( dropout) )
            if batch_norm is True:
                model.append( BatchNormalization() )
            model.append( Dense( n_neuron[i], activation=next(activation))  )
        model.append( Dense(n_output, activation=None) )

    def call( self, x, *args, training=False):
        if x.ndim == 4: #if we have the image data call then simply 
            x = args[0] #overwrite the image data with the features
        for layer in self.architecture:
            x = layer( x, training=training)
        return x



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
        super(ForwardNormalization, self).__init__( n_output)
        activation = Cycler( activation)
        self.architecture = []
        model = self.architecture
        for i in range( len(hidden_neurons) ):
            model.append( Dense( hidden_neurons[i], activation=next(activation) )  ) 
            ## dropout if specified
            if isinstance( dropout, float):
                model.append( Dropout( dropout) )
            elif isinstance( dropout, tuple) or isinstance( dropout, list):
                model.append( Dropout( dropout[i] ) )
            ## batch normalization if specified
            if batch_norm is True:
                model.append( BatchNormalization() )
            elif isinstance( batch_norm, tuple) or isinstance( batch_norm, list):
                if batch_norm[i] is True:
                    model.append( BatchNormalization() )

        model.append( Dense(n_output, activation=None) )

        
class ReconstructedANN( Model):
    """
    Model which served as a baseline which was solely reconstructed from the 
    paper data driven microstructure property relations (2018)
    """
    def __init__(self, n_output, n_neuron=[6,7], activation=['softplus','softplus'], *args, **kwargs): 
        super( ReconstructedANN, self).__init__( n_output, *args, **kwargs)
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



