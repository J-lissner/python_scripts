import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPool2D
from tensorflow.keras.layers import Conv2DTranspose, Layer
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import BatchNormalization, Dense, Concatenate, concatenate, Add
from tensorflow.python.ops import nn, nn_ops
from tensorflow.python.trackable.data_structures import ListWrapper
from math import ceil, floor



class LayerWrapper(ListWrapper):
  """
  This function is like a callable ListWrapper. It is used to store any
  type  of layer(of deep inception modules and generic layers).
  Each entry in the list (of the ListWrapper) is treated as one module. This
  module may contain a generic 'Layer', or another list, which then specifies
  the custom modules, i.e. resnet/inception modules. (which is a list of 
  layers/lists, each entry/sublist specifying a 'branch')
  It is able to consider deep inception modules but not nested inception modules. 
  Normal layers are given by layer classes, Inception modules are given by
  a list of layers, and deep inception modules are given by nested lists of layers.
  """
  def __init__( self, *args, **kwargs):
      super().__init__( list(args), **kwargs)

  def __call__( self, images, *layer_args, **layer_kwargs):
      """
      predict the current images using the layers with the <images> data.
      This function takes in any layers put into list up to 1 inception
      module (no nested inception modules) with arbitrary depth in each
      branch
      Parameters:
      -----------
      images:     tensorflow.tensor like
                  image data of at least 4 dimensions
      *layer_kw/args: 
                  additional inputs directy passed to each layer
      """
      for layer in self:
          if not isinstance( layer,list): #simple layer, or concatenator after inception module
              images = layer( images, *layer_args, **layer_kwargs) 
          else: #parallel conlvolution operations
              inception_pred = []
              for layer in layer:
                  if not isinstance( layer, list): #only single parallel convolutions
                      pred = layer(images, *layer_args, **layer_kwargs) 
                      inception_pred.append(pred)
                  else: #deep inception module
                      inception_pred.append( layer[0](images, *layer_args, **layer_kwargs) )
                      for deeper_layer in layer[1:]:
                          inception_pred[-1] = deeper_layer(inception_pred[-1], *layer_args, **layer_kwargs )
              images = inception_pred
      return images

  def freeze( self, freeze=True):
    for layer in self:
        if isinstance( layer, LayerWrapper):
            layer.freeze( freeze) 
        elif isinstance( layer, list):
            for sublayer in layer:
              if isinstance( sublayer, list):
                for subsublayer in sublayer:
                  subsublayer.trainable = not freeze
              else:
                sublayer.trainable = not freeze
        else:
            layer.trainable = not freeze




#### Single layers 
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
        if 'dilation_rate' in kwargs:
            self.k_size =  self.k_size + (self.k_size -1) * (kwargs['dilation_rate']-1)

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


class Conv2DTransposePeriodic( Conv2DTranspose):
    def __init__( self, *args, **kwargs):
        kwargs['padding'] = 'same' #enforce no padding and do the stuff by myself
        super(Conv2DTransposePeriodic, self).__init__( *args, **kwargs)
        if 'kernel_size' in kwargs:
            kernel_size = kwargs['kernel_size']
        else: 
            kernel_size = args[1]
        if 'strides' in kwargs:
            stride = kwargs['strides']
        else: 
            stride = args[2] 
        self.p_size = kernel_size//stride #might be a little excessive

    def __call__(self, data, *args, **kwargs):
        """
        shadow the original call, simply pad before upsampling and then 
        cut away the excessive part
        """
        upsampled = super().__call__( upsampling_padding( self.p_size, data ), *args, **kwargs)
        idx = max(2*self.p_size, 2)
        return upsampled[:, idx:-idx, idx:-idx ]


#### required functions (padding for now
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
    image_dim = len( data.shape) - 2 #n_smaples, n_channels
    if isinstance( kernel_size, int):
        kernel_size = image_dim*[kernel_size]
    for i in range( image_dim):
        left_pad =  kernel_size[i]//2 #start of axis
        right_pad = left_pad - 1 if (kernel_size[i] %2) == 0 else left_pad #end of axis
        ## left and right refers to where its 'stitched on', i.e. slices from the opposite side
        if right_pad == 0:
            end_pad = []
        else:
            end_slice = (i+2)*[slice(None)]
            end_slice[i+1] = slice( right_pad)
            end_slice = tuple( end_slice)
            end_pad = [data[end_slice]]
        start_slice = (i+2)*[slice(None)]
        start_slice[i+1] = slice( -left_pad, data.shape[i+1])
        start_slice = tuple( start_slice)
        data = concatenate( [data[start_slice], data ] + end_pad, axis=i+1 )
    return data

def upsampling_padding( pad_size, data):
    """
    periodically pad the image before upsampling to enforce
    periodic adjacency correctness
    Parameters:
    -----------
    pad_size:       list of ints or int
                    size of the padding in each dimension
    data:           tensorflow.tensor
                    image data of at least 3 channels, n_sample x n_x x n_y
    Returns:
    --------
    padded_data:    tensorflow.tensor
                    the padded data by the minimum required width
    """
    ## for now i assume that on even sized kernels the result is in the top left
    if isinstance( pad_size, int):
        pad_size = 2*[pad_size]
    pad_l = pad_size[1]
    pad_u = pad_size[0]
    pad_r = pad_l
    pad_b = pad_u
    ## the subscript refer to where it is sliced off, i.e. placedo n the opposite side
    if pad_r == 0:
        top_pad = []
    else:
        top_pad = [concatenate( [data[:,-pad_r:, -pad_u:], data[:,-pad_r:,:], data[:,-pad_r:, :pad_b] ], axis=2 ) ]
    bot_pad = [concatenate( [data[:,:pad_l, -pad_u:], data[:,:pad_l,:], data[:,:pad_l, :pad_b] ], axis=2 ) ]
    data = concatenate( [data[:,:,-pad_u:], data, data[:,:,:pad_b] ], axis=2 )
    data = concatenate( top_pad + [data] + bot_pad, axis=1 )
    return data

### modules
class DeepInception( Layer):
  def __init__( self, n_channels=64, pooling='average', *args, **kwargs):
    """
    So the general idea is to have a reusable module with the deep inception modules
    I will start by just trying out a downsampling factor of 8 after each module
    With no fancy bypasses and just averagepooling and convolution
    General layout idea is to have different downsampling factors of average pooling
    and then whatever convolution operations
    Each branch has n_features channels, with a compression to n_features//2 channels before concatenation
    channels specifies if the channels should be linearly increased or constant within
    one module
    Parameters:
    -----------
    n_channels:     list of (up to 3) ints, default 64
                    number of channels defined for convolution, linearly interpolates 
                    as the number of channels in each branch for the first two ints
                    in the list. The last integer defines the number of output channels
                    after concatenation and 1x1 conv (list of len 2 is also allowed) 
    pooling:        str, default 'average'
                    pre-pooling method in each branch, choose between 'average' and 'max'
    """
    super().__init__( *args, **kwargs)
    ## input preprocessing
    n_out = [n_channels] if not hasattr( n_channels, '__iter__') else list( n_channels)
    n_out = 2*n_out if len( n_out) == 1 else n_out
    if pooling in ['average', 'avg']:
        pooling = AvgPool2DPeriodic
    elif not isinstance( pooling, Layer):
        pooling = MaxPool2DPeriodic
    ## function definition and abbreviations
    default_conv = lambda n_out=n_out[1], kernel_size=3, strides=1: Conv2DPeriodic( n_out, kernel_size, strides) 
    conv_1x1     = lambda n_out=n_out[1]: Conv2D( n_out, kernel_size=1, activation='selu' )
    n_channels   = lambda i, n_conv: n_out[0] + i/n_conv * (n_out[1] - n_out[0] )
    branch_initialization = lambda n: [ LayerWrapper() for _ in range( n)]
    ## hardwired parameters of the deep inception module
    self.n_branches = 4 
    self.branches = branch_initialization( self.n_branches)
    self.normalizers = branch_initialization( self.n_branches +1) #+ normalizer after concat
    self.concatenator = LayerWrapper( Concatenate())
    self.normalizers[-1].append( SqueezeExcite( n_out[-1], Conv2D( n_out[-1], kernel_size=1, activation='selu' )) )
    ## increasinlgy higher coarse graining branches
    # 6 conv, 3 1x1; first branch
    n_op = 6 #number of actual convolution operations
    branch = self.branches[0]
    branch.append( default_conv(  n_out=n_channels(1, n_op), strides=2) )
    branch.append( default_conv(  n_out=n_channels(2, n_op)) )
    branch.append( conv_1x1(      n_out=n_channels(2, n_op)) )
    branch.append( default_conv(  n_out=n_channels(3, n_op), strides=2) )
    branch.append( default_conv(  n_out=n_channels(4, n_op)) )
    branch.append( conv_1x1(      n_out=n_channels(4, n_op)) )
    branch.append( default_conv(  n_out=n_channels(5, n_op), strides=2) )
    self.normalizers[0].append( SqueezeExcite( n_channels(6, n_op), default_conv( n_out=n_channels(6, n_op))) )
    #pool+ 1 1x1 + 4 conv operations; second branch
    n_op = 4
    branch = self.branches[1]
    branch.append( pooling( 2) ) 
    branch.append( default_conv(  n_out=n_channels(1, n_op), kernel_size=5, strides=2) )
    branch.append( default_conv(  n_out=n_channels(2, n_op)) )
    branch.append( conv_1x1(      n_out=n_channels(2, n_op)) )
    branch.append( default_conv(  n_out=n_channels(3, n_op), kernel_size=5, strides=2) )
    self.normalizers[1].append( SqueezeExcite( n_channels(4, n_op), default_conv( n_out=n_channels(4, n_op)) ) )
    #pool + 1 1x1 + 3conv
    n_op =  3
    branch = self.branches[2]
    branch.append( pooling( 4) )
    branch.append( default_conv(  n_out=n_channels(1, n_op), kernel_size=5, strides=2) )
    branch.append( default_conv(  n_out=n_channels(2, n_op), kernel_size=5) )
    branch.append( conv_1x1(      n_out=n_channels(2, n_op) ) )
    self.normalizers[2].append( SqueezeExcite( n_channels(3, n_op), default_conv( n_out=n_channels(3, n_op)) ) )
    #pool + 1 1x1 + 3 conv
    n_op = 3
    branch = self.branches[3]
    branch.append( pooling( 8) )
    branch.append( default_conv(  n_out=n_channels(1, n_op), kernel_size=5 ) )
    branch.append( default_conv(  n_out=n_channels(2, n_op), kernel_size=5 ) )
    branch.append( conv_1x1(      n_out=n_channels(2, n_op) ) )
    self.normalizers[3].append( SqueezeExcite( n_channels(3, n_op), default_conv( n_out=n_channels(3, n_op), kernel_size=5 )  ) )


  def freeze( self, freeze=True):
      for i in range( self.n_branches):
          self.branches[i].freeze( freeze)
          self.normalizers[i].freeze( freeze)
      self.normalizers[-1].freeze( freeze)
      self.concatenator.freeze( freeze)

  def __call__( self, images, x_extra=None, training=False):
      """
      Predict the deep inception module. May also take an extra feature
      vector which will be (i think) added to the squeeze and excitation bypass
      Parameters:
      -----------
      images:       tensorflow.tensor of shape [n, *res, n_channels]
                    input data to the module
      x_extra:      tensorflow.tensor of shape [n, n_fts]
                    feature vector to be added to the SnE block
      training:     bool, default False
                    flag to inform the ANN whether its training
      Returns:
      --------
      images:       tensorflow.tensor of shape [n, *res, n_channels]
                    output of the module, downsampled by factor 8
      """
      prediction = []
      for i in range( self.n_branches ):
          prediction.append( self.branches[i]( images, training=training ) )
          prediction[-1] = self.normalizers[i]( prediction[-1], x_extra=x_extra, training=training)
      prediction = self.concatenator( prediction, training=training )
      return self.normalizers[-1]( prediction, x_extra=x_extra, training=training )


class ModularizedDeep( Layer):
    def __init__( self, n_conv=8, pooling=8, n_channels=64, bypass=False, pool_type='max', *args, **kwargs ):
        """
        Initialize a deep inception module with multiple branches of 
        of different receptive field. The module is only implemented
        for the following pooling values: 'pooling in [4,6,8,12,16]'
        The higher the pooling, the more branches will be used
        The number of branches is hardwired and linked to the pooling 
        value
        On channel unition, the channels are normalized with the 
        'NormalizationLayer', and summed up. Thereafter they are
        normalized again.
        Deploys periodic padding in each layer
        Parameters:
        -----------
        n_conv:     int, default 8
                    maximum number of convolution operations
        pooling:    int, default 8
                    pooling factor of the module
        n_channels: list of two ints, default [64,64]
                    number of channels in the module, linearly scales
                    from [a,b] in each branch, the last number is the
                    number of output channels
        bypass:     operation, default False
                    if given, it is the unition operation (i.e. concat/add)
                    of the bypass which uses 'pool_type' and the unition 
                    passed after the bypass anotehr normalization layer 
                    is conducted
        pool_type:  str or tensorflow.keras.layer, default 'max'
                    pool operation or type as string to preceede in each branch 
        """
        ##input preprocessing and alias allocation
        super().__init__( *args, **kwargs)
        n_channels = [n_channels] if not hasattr( n_channels, '__iter__') else list( n_channels)
        n_channels = 2*n_channels if len( n_channels) == 1 else n_channels
        conv = lambda i, pooling: Conv2DPeriodic( 
                int(n_channels[0]+(i)/n_conv*(n_channels[1]-n_channels[0])), 
                kernel_size=3, strides=pooling, activation='selu'  )
        if isinstance( pool_type, str):
            if 'avg' in pool_type.lower() or 'average' in pool_type.lower():
                pool_operation = AvgPool2DPeriodic
            else: #just default to maxpooling in any other case
                pool_operation = MaxPool2DPeriodic
        else:
            pool_operation = pool_type
        ## hardwired branches given the specified pooling
        if pooling not in [4,6,8,12,16]:
            raise ValueError( 'wrong parameter set for pooling in my_layers.ModularizedDeep')
        if pooling == 4: ##hardwired factorization for each parameters
            factorization = [[None,2,2], [2,2] ] 
        elif pooling == 6:
            factorization = [[None,2,3], [2,3], [3,2]]
        elif pooling == 8:
            factorization = [[None,2,2,2], [2,2,2], [4,2]]
        elif pooling == 12:
            factorization = [[None,2,2,3], [2,2,3], [4,3], [6,2]]
        elif pooling == 16:
            factorization = [[None,2,2,2,2], [2,2,2,2], [4,2,2], [8,2]] 
        factorization.append( [pooling]) #last layer with full pooling 
        print( f'building a deep inception module with pooling={pooling} and {len(factorization)} branches') 
        strides = [] #branch information
        for i, stride_split in enumerate( factorization):
            branch = [ stride_split.pop(0)]
            n_free = n_conv - len( stride_split) 
            branch = branch + n_free*[1] + stride_split
            strides.append( branch)
        self.conv_layers = []
        self.n_branches = 0
        for branch in strides:
            self.n_branches += 1
            branch_layers = LayerWrapper( )
            for i, pooling in enumerate( branch):
                if i == 0 and pooling is not None:
                    branch_layers.append( pool_operation( pooling))
                elif pooling is not None:
                    branch_layers.append( conv( i, pooling) )
            branch_layers.append( NormalizationLayer( n_channels[-1])  )
            self.conv_layers.append( branch_layers )
        #self.conv_layers.append( Add())
        self.conv_layers.append( NormalizationLayer( n_channels[-1]) )
        ## and i think the upper part thats it
        self.bypass = None
        if bypass:
            self.bypass = pool_operation( pooling)
            #do bypass operation here
            self.merge = LayerWrapper(  )
            self.merge.append( NormalizationLayer( n_channels[-1] ) )



    def __call__( self, images, x_extra=None, *args, training=False, **kwargs):
        """
        """
        ### module evaluation
        features = 0
        for layer in self.conv_layers[:-1]: #everything except the normalization layer
            features += layer( images, training=training, *args, **kwargs) 
        features = self.conv_layers[-1]( features) #1x1 + add is equivalent to concat + 1x1 but more memory friendly
        if self.bypass:
            bypass = self.bypass( images, training=training, *args, **kwargs)
            features = self.merge( [bypass,images], training=training, *args, **kwargs)
        return features

    
    def freeze( self, freeze=True):
        for layer in self.conv_layers:
            try:
                layer.freeze( freeze)
            except:
                layer.trainable = not freeze
        if self.merge:
            self.merge.freeze( freeze)



class SqueezeExcite(Layer):
    """
    get the Squeeze and excitation block for any layer, or callable. Only
    works in the 2D convolutional neural netowrk context (since globalPool2D
    is used)
    It is basically a reweighting of channels using the global information, as
    well as each channel for information input. The main difference to 1x1 conv 
    is that it returns channel wise weights from 0-1 (sigmoid), instead of a
    sum of all channels in each channel. So it weights each individual output
    channel instead of weighting each individual input channel over all output channels
    This has added functionality to the literature squeeze and excitation 
    module, namely there can be a layer in parralel to the squeeze and excitation
    sidepass
    """
    def __init__( self, n_channels, layer=None, reduction_ratio=16, *args, **kwargs):
        """
        Implement the defaut squeeze and excitation network
        Parameters:
        -----------
        n_channels:     int or list of ints
                        [(input), output ] n_channels of the layer, if an 
                        int it assumes that input == output
        layer:          tensorflow.keras.layer or callable
                        callable which to squeeze and excite
        reduction_rate: int, default 16
                        reduction rate of the channels 
        """
        ## input processing
        super().__init__( *args, **kwargs)
        n_channels = [n_channels] if not hasattr( n_channels, '__iter__') else list( n_channels)
        n_channels = 2*n_channels if len( n_channels) == 1 else n_channels
        reduced_representation = max( n_channels[0]//reduction_ratio, 4)
        layer = (lambda x, *args, **kwargs: x)  if layer is None else layer
        self.pooling = GlobalAveragePooling2D()
        self.se_block = LayerWrapper()
        self.se_block.append( Dense( reduced_representation, activation='selu') )
        self.se_block.append( Dense( n_channels[1], activation='sigmoid' ))
        self.layer = layer

    def __call__( self, images, *channel_args, x_extra=None, training=False, **channel_kwargs):
        weights = self.pooling( images)
        if x_extra is not None:
            weights = concatenate( [weights, x_extra] )
        weights = self.se_block( weights, training=training)
        weights = tf.reshape( weights, [weights.shape[0], 1,1, weights.shape[-1] ] )
        return weights * self.layer( images, *channel_args, training=training, **channel_kwargs) 

    def freeze( self, freeze=False):
        self.se_block.freeze( freeze)
        try: 
            self.layer.freeze( freeze) #layerwrapper
        except:
            self.layer.trainable = not freeze


class NormalizationLayer( SqueezeExcite):
    """
    basically the same as the Squeeze and Excite block above, but there is a
    default 1x1 conv at the same time to the SnE bypass
    """
    def __init__( self, n_channels, reduction_ratio=16, *args, **kwargs):
        """
        Implement the squeeze and excitation block with parallel 1x1 conv
        Parameters:
        see SqueezeExcite.__init__ 
        """
        n_channels = 2*[n_channels] if isinstance( n_channels,int) else n_channels
        bypass_layer = Conv2D( n_channels[1], kernel_size=1, activation='selu' )
        super().__init__( n_channels, layer=bypass_layer, reduction_ratio=reduction_ratio, *args, **kwargs)



class InceptionModule():
  def __init__( self, n_out=32, n_branch=12, downsample=4, max_kernel=5, pooling='average', activation='selu'):
    """
    Define an inception module which has a 1x1 bypass and multiple two deep
    convolutional branches. The number of conovlutional branches is computed
    by <(max_kerrnel-1)//2>, since it takes them of a 2 increment.
    The inception module achieves downsampling through stride and <pooling>
    the 1x1 bypass. Always takes the periodically padded layers.
    Parameters:
    -----------
    n_out:          int, default 32
                    number of output channels of the inception module
    n_branch:       int, default 12
                    number of channels per branch in inception module
    downsample:     int, default 4
                    downsampling, must be power of 2
    max_kernel:     int, default 5
                    maximum size of the second kernel in the layer
                    preferably an odd number
    pooling:        string, default 'average'
                    max or average pooling conducted before the 1x1 bypass
    activation:     string, default 'selu'
                    activation function at each convolutional kernel
    """
    ## input preprocessing and variable allocation
    n_conv = (max_kernel-1)//2
    self.inception = []
    ## 1x1 bypass and concatenation
    bypass = []
    concat = [Concatenate()]
    if 'average' in pooling:
        bypass.append( AvgPool2DPeriodic( downsample)  )
    elif 'max' in pooling:
        bypass.append( MaxPool2DPeriodic( downsample)  )
    bypass.append( Conv2D( filters=n_branch, kernel_size=1, strides=1, activation=activation) )
    concat.append( Conv2D( filters=n_out, kernel_size=1, strides=1, activation=activation) )
    concat.append( BatchNormalization() )
    ## different convolution branches
    for i in range( n_conv):
        self.inception.append( [Conv2DPeriodic( filters=n_branch, kernel_size=3, strides=min(downsample,2), activation=activation ) ])
        k = 1
        while 2*k/downsample < 1 or k < i:
          kernel_size = min(k,i)*2 + 3 
          if downsample/(k*2) > 1:
              stride = 2
          else: 
              stride = 1
          self.inception[i].append( Conv2DPeriodic( filters=n_branch, kernel_size=kernel_size, strides=stride, activation=activation ) )
          k += 1
    ## concatenation and 1x1 convolution 
    self.inception.insert( 0, bypass)
    self.inception.append( concat )


  def __call__( self, images, training=False, *args, **kwargs):
    """ 
    evaluate the inception module, simply evaluate each branch and
    keep the output of each branch in memory until concatenated 
    """
    x = []
    for i in range( len( self.inception) -1 ):
        for j in range( len( self.inception[i]) ):
            if j == 0:
                x.append( self.inception[i][j]( images, training=training) )
            else:
                x[i] = self.inception[i][j]( x[i], training=training ) 
    #concatenation and 1x1 convo
    for layer in self.inception[-1]:
        x = layer( x, training=training)
    return x

