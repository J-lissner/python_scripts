import tensorflow as tf
from tensorflow.keras import Model 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate, Layer, concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, GlobalAveragePooling2D
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, Conv2DTransposePeriodic
from literature_layers import UresnetDecoder, UresnetEncoder, MsPredictor
from fully_conv import MultilevelNet

class UResNet(Model, MultilevelNet):
    ## model applied in my field: https://www.sciencedirect.com/science/article/abs/pii/S0309170819311145
    ## originally from: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8309343
    """ 
    Inherits from my multilevel net though it does not fully support functionaly
    I will write a workaround to match the code template
    Does not implement any freezing even though it has freeze functions
    CARE: this is very 'fragile'
    """
    def __init__( self, n_output, n_levels=3, n_branches=1, n_channels=15, channel_function=None, *args, **kwargs  ):
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
            channel_function = lambda i, n_channel: 2**(i) * n_channel
        self.n_levels = n_levels
        self.decoder = []
        self.encoder = []
        self.concatenator = Concatenate()
        ## build as many decoders as we have niputs
        for j in range( n_branches):
          self.encoder.append( [ UresnetEncoder( n_channels, first=True) ] ) 
          for i in range( n_levels): 
            self.encoder[-1].append( UresnetEncoder( channel_function(i, n_channels) ) )
        if n_branches == 1:
            self.encoder = self.encoder[0]
        ## build the single decoder
        for i in range( n_levels): 
            self.decoder.append( UresnetDecoder( channel_function(i, n_channels) ) ) 
        self.decoder = self.decoder[::-1]
        self.predictor = Conv2D( n_output, kernel_size=1, strides=1)

    def call( self, images, level=None, training=False):
        """
        Parameters:
        -----------
        images:     input images
        trainin:    bool, used for the layers
        level:      None, just here to implement the modality of the model
        """
        ## first procesing + downward path
        if images.shape[-1] > 1:
            levels = []
            for i in range( images.shape[-1] ):
              levels.append( [ self.encoder[i][0]( images[...,i][...,tf.newaxis], training=training) ] )
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
            
    ## abstractmethods workaround to match the code template
    def predictor_features( self, *args):
        """ is required for the multievel setup which this one does not support"""
        return list( args) #must me list 
    def predict_tip( self, *args, **kwargs):
        return self( *args, **kwargs) 
    ## functions without any functionality
    def freeze_upto( self, *args, **kwargs):
        pass 
    def freeze_predictor( self, *args, **kwargs):
        pass 
    



class MSNet(Model, MultilevelNet):
    ## model in my field: https://link.springer.com/article/10.1007/s11242-021-01617-y
    ## originally from https://openaccess.thecvf.com/content_ICCV_2019/papers/Shaham_SinGAN_Learning_a_Generative_Model_From_a_Single_Natural_Image_ICCV_2019_paper.pdf
    """
    For now the functionality of the multilevel net is not implemented
    """
    def __init__( self, n_output, n_levels=4, n_channels=2, channel_function=None, n_branches=None, *args, **kwargs):
        """
        n_branches is just a parameter to enable modality
        """
        super().__init__( n_output, *args, **kwargs )
        if channel_function is None:
            channel_function = lambda i, n_channel: n_channel**(2*i+1)
        self.n_levels = n_levels
        self.architecture = [ MsPredictor( n_channels, n_output )]
        self.poolers = [ lambda x, *args, **kwargs: x]
        model = self.architecture
        for i in range( n_levels):
            model.append( MsPredictor( channel_function(i, n_channels), n_output ))
            self.poolers.append( AvgPool2DPeriodic( 2**(i+1)) )
        self.architecture = self.architecture[ ::-1]
        self.poolers = self.poolers[ ::-1]
    
    def call( self, images, level=False, training=False):
        """
        Yield the models prediction on the original resolution
        With <level> one  can specify which levels are requested, for training
        in the literature every level was trained simultaneously
        """
        level = range(self.n_levels + 1) if level is True else level 
        level = [level] if not (isinstance( level, bool) or hasattr( level, '__iter__' ) ) else level
        predictions = []
        prediction = self.architecture[0]( self.poolers[0]( images) ) 
        ## check if first level is desired
        if level is not False and 0 in level: 
            predictions.append( prediction)
            if 0 == max( level):  
                return predictions[0]
        ## go up every level
        for i in range( 1, self.n_levels+1):
          prediction = self.architecture[i]( self.poolers[i]( images), prediction, training=training ) 
          ## add current level to list if asked for
          if level is not False and i in level: 
            predictions.append( prediction)
            if i == max( level):  
                if len( level) == 1: return predictions[0]
                else: return predictions 
        ## check if only last level is requested
        if level is not False:  #may only happen if last level requested
            if len( level) == 1: return predictions[0]
            else: return predictions
        return prediction
        

    ## abstractmethods workaround to match the code template
    def predictor_features( self, images):
        """ is required for the multievel setup which this one does not support"""
        previous_prediction = self( images, level=self.n_levels-1)
        return [ images, previous_prediction]

    def predict_tip( self, images, previous_prediction, training=False, **kwargs):
        prediction = self.architecture[-1]( images, previous_prediction, training=training)
        return self( *args, **kwargs) 

    ## functions without any functionality
    def freeze_upto( self, freeze_limit, freeze=True):
        for i in range( freeze_limit+1):
            self.architecture[i].freeze( freeze)

    def freeze_layers( self, freeze=True):
        for i in range( self.n_levels):
            self.architecture[i].freeze( freeze)
        self.freeze_predictor( freeze)

    def freeze_predictor( self, freeze=True, **kwargs):
        """ required for unfreezing the predictor during only predictor training """
        self.architecture[-1].freeze( freeze)
    


