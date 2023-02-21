import tensorflow as tf
from my_models import Model #tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate, Layer, concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, GlobalAveragePooling2D
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, Conv2DTransposePeriodic
from literature_layers import UresnetDecoder, UresnetEncoder, MsPredictor

class UResNet(Model):
    ## model applied in my field: https://www.sciencedirect.com/science/article/abs/pii/S0309170819311145
    ## originally from: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8309343
    def __init__( self, n_output, n_levels=3, n_branches=1, n_channels=15, channel_function=None *args, **kwargs  ):
        """
        Parameters:
        -----------
        n_output:   int
                    number of channels to predict
        n_levels:   int
                    number of levels, i.e. how often to downsample
        n_channels: int
                    how many channels the first layer has
        channel_function:   callable with i, n_channel input
                            how many channels on lower levels

        """
        super().__init__( n_output, *args, **kwargs )
        if channel_function is None:
            channel_function = lambda i, n_channel: 2**(i+1) * n_channel
        self.n_levels = n_levels
        self.decoder = []
        self.encoder = []
        self.concatenator = Concatenate()
        ## build as many decoders as we have niputs
        for j in range( n_branches):
          self.encoder.append( [ UresnetEncoder( n_channels, first=True) ] ) 
          for i in range( n_levels): 
            self.encoder[-1].append( UresnetEncoder( channel_function(i, n_channel) ) )
        if n_branches == 1:
            self.encoder = self.encoder[0]
        ## build the single decoder
        for i in range( n_levels): 
            self.decoder[-1].append( UresnetDecoder( channel_function(i, n_channel) ) ) 
        self.decoder = self.decoder[::-1]
        self.predictor = Conv2D( n_output, kernel_size=1, strides=1)

    def call( self, images, training=False):
        ## first procesing + downward path
        if images.shape[-1] > 1:
            levels = []
            for i in range( images.shape[-1] ):
              levels.append( [ self.encoder[i]( images[...,i], training=training) ] )
                  for layer in self.encoder[i][1:]:
                    levels[-1].append( layer( levels[i][-1], training=training) ) 
            levels = [ self.concatenator(x) for x in zip( *levels)]
        else: #only single feature
            levels = [ self.encoder[0](images, training=training) ] 
            for encoder in self.encoder[1:]:
                levels.append( encoder( levels[-1], training=training ) )
        ### upward path which also clears the down memory
        for decoder in self.decoder:
            #if self.predict_sides: predictions.append( side_predictor[i](levels[-1] ))
            levels[-1] = decoder( levels.pop( -1), levels[-1], training=training )
        ### final prediction
        return self.predictor( levels[-1], training=training)
            

class MSNet(Model):
    ## model in my field: https://link.springer.com/article/10.1007/s11242-021-01617-y
    ## originally from https://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf
    def __init__( self, n_output, n_levels=4, base_channels=2, channel_function=None, *args, **kwargs):
        super().__init__( n_output, *args, **kwargs )
        if channel_function is None:
            channel_function = lambda i, n_channels: n_channels**(2*i+1)
        self.n_levels = n_levels
        self.architecture = [ MsPredictor( base_channels, n_output )]
        self.poolers = [ lambda x, *args, **kwargs: x]
        model = self.architecture
        for i in range( n_levels):
            model.append( MsPredictor( channel_function(i), n_output ))
            self.poolers.append( AvgPool2DPeriodic( 2**(i+1)) )
        self.architecture = self.architecture[ ::-1]
        self.poolers = self.poolers[ ::-1]
    
    def call( self, images, training=False):
        """
        Yield the models prediction on the original resolution
        If <training> is True, it returns a list of predictions where
        each sublevels prediction is given. The original scale is in the
        last slot
        """
        prediction = self.architecture[0]( self.poolers[0]( images) )
        if training is True:
          prediction = [ prediction]
          for i in range( 1, self.n_levels+1):
            prediction.append( self.architecture[i]( self.poolers[i]( images), prediction[-1], training=training ) )
        else:
          for i in range( 1, self.n_levels+1):
            prediction = self.architecture[i]( self.poolers[i]( images), prediction ) 
        return prediction
        

