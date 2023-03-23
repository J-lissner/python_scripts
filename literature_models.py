import tensorflow as tf
import numpy as np
from tensorflow.keras import Model 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate, Layer, concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, GlobalAveragePooling2D
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, Conv2DTransposePeriodic
from literature_layers import UresnetDecoder, UresnetEncoder, MsPredictor
from fully_conv import MultilevelNet


class InceptionNet( Model):
    def __init__( self, n_out, channel_reduction=2, ):
        """
        Build the inception net as is in the literature (szegedy2015going)
        For now i will jsut brute force the number of channels
        """
        super().__init__( n_out, *args, **kwargs )
        n_3 = lambda n, factor=2: [n, factor*n]
        inception_channels = []
        inception_channels.append( [ 64, 128, 192, 160, 128, 112, 256, 256, 384])  #1x1 bypass
        inception_channels.append( [ [64, 192], [96, 128], [96,208], [112, 224], [128, 256], [144,288], [160,320],
                                      [160, 320], [192,384] ] ) #1x1->3x3
        inception_channels.append( [ [16,32], [32,96], [16,48], [24,64], [24,64], [32, 64], [32, 128], [32, 128], [48,128] ]) #1x1->5x5
        inception_channels.append( [32, 64, 64, 64, 64, 64, 128, 128, 128] ) #pool -> 1x1
        inception_channels = [ (np.array(x)/channel_reduction).astype(int) for x in inception_channels)
        n_channels = [64, 64, 192] #for the three preceeding/postceeding operations
        dense_neurons = 1000 #keep it fixed, since there memory is ok
        side_channels =  128 
        n_channels = (np.array( n_channels )/channel_reduction).astype(int) #first 
        print( [len(x) for x in inception_channels] )
        
        ## split in three parts for implementation. After each part the sidepredictor is 
        self.first_part = [Conv2DPeriodic( n_channels[0], kernel_size=7, strides=2)] 
        self.first_part.append( MaxPool2DPeriodic( 3, strides=2) )
        self.first_part.append( tf.nn.local_response_normalization)
        self.first_part.append( Conv2D( n_channels[1], kernel_size=1 ) )
        self.first_part.append( Conv2DPeriodic( n_channels[2], kernel_size=3 ) )
        self.first_part.append( tf.nn.local_response_normalization)
        self.first_part.append( MaxPool2DPeriodic( 3, strides=2) )
        for i in range( 3): #first three inception modules
            n_current = [ x[i] for x in inception_channels]
            self.first_part.append( InceptionModule( n_current) )
            if i == 1:
                self.first_part.append( MaxPool2DPeriodic( 3, strides=2) )
        ## now comes the first branch off during training
        self.second_part = []
        for i in range( 3, 3+3): #next three inception modules
            n_current = [ x[i] for x in inception_channels]
            self.second_part.append( InceptionModule( n_current) )
        self.third_part = []
        for i in range( 6, 6+3): #next three inception modules
            n_current = [ x[i] for x in inception_channels]
            self.third_part.append( InceptionModule( n_current) )
            if i == 6:
                self.third_part.append( MaxPool2DPeriodic( 3, strides=2) )
        self.third_part.append( AvgPool2DPeriodic( 7, strides=7) )
        self.third_part.append( Flatten())
        self.third_part.append( Dropout(0.4))
        self.third_part.append( Dense( dense_neurons, activation='selu') )
        self.third_part.append( Dense( n_out, activation='none') )

        self.side_predictors = [ [] []]
        for i in range( 2): #both are the same
            self.side_predictors[i].append( AvgPool2dPeriodic( 5, strides=3))
            self.side_predictors[i].append( Conv2D( side_channels, strides=3))
            self.side_predictors[i].append( Flatten() )
            self.side_predictors[i].append( Dense( dense_neurons, activation='selu' ) )
            self.side_predictors[i].append( Dropout(0.7) )
            self.side_predictors[i].append( Dense( n_out, activation='none' ) )

    def call( self, images, side_prediction=False, training=False):
        images = self.first_part( images, training=training)
        if side_prediction:
            side_predictions = [ self.side_predictors[0]( images, training=training) ]
        images = self.second_part( images, training=training)
        if side_prediction:
            side_predictions.append( self.side_predictors[1]( images, training=training) )
        images = self.third_part( images, training=training)
        if side_prediction:
            return images, side_predictions #also side predictions for loss
        return images #return full prediction




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
            if len( level) == 1:  return predictions[0] #same as [-1]
            elif i == max(level): return predictions
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
    


