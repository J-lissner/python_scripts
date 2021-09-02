import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AveragePooling2D, Flatten, GlobalAveragePooling2D

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


