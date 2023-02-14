import tensorflow as tf
from my_models import Model #tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, GlobalAveragePooling2D
from my_layers import Conv2DPeriodic, AvgPool2DPeriodic, MaxPool2DPeriodic, Conv2DTransposePeriodic
from literature_layers import UresnetDecoder, UresnetEncoder, MsPredictor


class UResNet(Model):
    def __init__( self, n_output, n_levels=3, base_channels=15, *args, **kwargs  ):
        super().__init__( n_output,n_output,  *args, **kwargs )
        self.n_levels = n_levels
        down_channels = lambda i: 2**(i+1) * base_channels
        up_channels = lambda i: 2**(n_levels-(i+1) ) * base_channels
        self.encoder = [ UresnetEncoder( base_channels, first=True) ]
        self.decoder = []
        for i in range( n_levels):
            self.encoder.append( UresnetEncoder( down_channels(i) ) )
            self.decoder.append( UresnetDecoder( up_channels(i) ) ) 
        self.predictor = Conv2D( n_output, kernel_size=1, strides=1)

    def call( self, images, training=False):
        ## first procesing + downward path
        levels = [ self.encoder[0](images, training=training) ] 
        for encoder in self.encoder[1:]:
            levels.append( encoder( levels[-1], training=training ) )
        ### upward path which also clears the down memory
        for decoder in self.decoder:
            levels[-1] = decoder( levels.pop( -1), levels[-1], training=training )
        ### final prediction
        return self.predictor( levels[-1], training=training)
            

class MSNet(Model):
    def __init__( self, n_output, n_levels=4, base_channels=2, *args, **kwargs):
        super().__init__( n_output, *args, **kwargs )
        n_channels = lambda i: base_channels**(2*i+1)
        self.n_levels = n_levels
        self.architecture = [ MsPredictor( base_channels, n_output )]
        self.poolers = [ lambda x, *args, **kwargs: x]
        model = self.architecture
        for i in range( n_levels):
            model.append( MsPredictor( n_channels(i), n_output ))
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
            prediction.append( self.architecture[i]( self.poolers[i]( images), prediction[-1] ) )
        else:
          for i in range( 1, self.n_levels+1):
            prediction = self.architecture[i]( self.poolers[i]( images), prediction ) 
        return prediction
        


            


class UResNetParallel(Model):
    """ this one should be the model from the pore flow paper 
    https://www.sciencedirect.com/science/article/abs/pii/S0309170819311145
    It just hurts too much to implement that one, so i won't finish it.
    Especially since the input features don't neccessarily make sense.
    For now i willj ust use the deep uresnet
    """
    def __init__( self, n_output, n_input=1, n_levels=3, base_channels=15, *args, **kwargs  ):
        super().__init__( n_output, *args, **kwargs )
        self.n_levels = n_levels
        self.n_input = n_input
        down_channels = lambda i: base_channels*(i+2)
        up_channels = lambda i: (n_levels-(i+1) ) * base_channels
        self.encoder = [ self.n_input*[UresnetDecoder( base_channels, first=True) ] ]
        self.decoder =  []  
        for i in range( n_levels):
            for i in range( n_input):
                self.encoder[i].append( UresnetEncoder( down_channels(i) ) )
            self.decoder.append( UresnetDecoder( up_channels(i) ) )
        self.concatenator = Concatenate() 
        self.predictor = Conv2D( n_output, kernel_size=1, strides=1)

    def call( self, images, training=False):
        ## first procesing + downward path
        levels = []
        for i in range( self.n_input): 
            levels.append( [ self.encoder[0](images) ]  )
            for encoder in self.encoder[i][1:]:
                levels[i].append( encoder( levels[-1] ) )
        for i in range( len( levels[0]) ):
            levels[i] = self.concatenator( levels[i])
        ### upward path which also clears the down memory
        for i, decoder in zip( range( self.n_levels), self.decoder):
            levels[-1] = decoder( levels.pop( -1), levels[-2] )
        ### final prediction of the last remaining level
        return self.predictor( levels[-1])




class AlexNet(Model):
    def __init__( self, output_size ):
        """
        set up the alex net (which might not be 100% alex net but really close to it)
        """
        super(AlexNet, self).__init__()
        self.architecture = [] #abbreviation for self.architecture
        model = self.architecture
        ## feature extraction
        model.append( Conv2D( 64, kernel_size=(11,11), strides=(4,4), padding='same', activation='relu' ))
        model.append( MaxPool2D( pool_size=(3,3), strides=(2,2) ))
        model.append( Conv2D( 192, kernel_size=(5,5), strides=(1,1), padding='same', activation='relu' ))
        model.append( MaxPool2D( pool_size=(3,3), strides=(2,2) ))
        model.append( Conv2D( 384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu' ))
        model.append( Conv2D( 256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu' )) 
        model.append( MaxPool2D( pool_size=(3,3), strides=(2,2) ) )
        ## pooling
        model.append( AveragePooling2D(3,3) ) 
        ## regressor
        model.append( Flatten() )
        model.append( Dropout( 0.5) )
        model.append( Dense( 4096, activation='relu') )
        model.append( Dropout( 0.5) )
        model.append( Dense( 4096, activation='relu') )
        model.append( Dense( output_size) )
        

    def call(self, x, training=False):
        for layer in self.architecture:
            x = layer( x, training=training)
        return x

    def predict( self, x):
        return self( x, training=False )


