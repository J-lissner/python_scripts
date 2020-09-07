import tensorflow as tf
from tensorflow.keras.layers import Layer

def L_Norm(Layer):
    """
    This layer currently always outputs 1 value.
    It can be set to be the minimum or maximum of outputs with norm='-inf' or norm='inf'.
    For a smoother version, the norm can be set to an integer to get the L_n -norm
    """
    def __init__(self, norm=20, num_outputs=1): #TODO possibly implement num_outputs if norm is an int
        super( LNorm, self).__init__()
        if not ( norm in {'inf', '-inf'} or type(norm) == int):
            raise Exception( "Accepted norms are any integer or 'inf' and '-inf', refering to max and min")
            #specify the type of norm, careful "inf" and "-inf" did on the toy case not yield any appropriate results
        self.num_outputs=1
        self.norm = norm

    def build(self, input_shape):
        # initialize the connection the the next layer. The weights are set to one that they can sum up the inputs
        self.kernel = self.add_weight( name='comparator', shape=[int(input_shape[-1]),self.num_outputs], trainable=False)
        self.kernel.assign( np.ones( [int(input_shape[-1]),self.num_outputs]) )

    def call(self, in_array):
        if self.norm == 'inf':
            return tf.math.reduce_max(in_array)
        elif self.norm == '-inf':
            return tf.math.reduce_min(in_array)
        else:
            return  (tf.matmul( tf.abs(in_array**self.norm), self.kernel ))**(1/self.norm)
            # this is basically L-"norm" norm
~
