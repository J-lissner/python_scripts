import tensorflow as tf

#@tf.function decorator not written here, because it does not allow to train multiple ANN
def train_step(x, y, model, loss_object):
    """
    Compute the gradients of the loss w.r.t. the trainable variables of the model
    Input:
        x           training data 
        y           training target
        model       ANN, tf.keras.Model
        loss_object loss from tf.keras.losses (or self defined)
    """
    with tf.GradientTape() as tape:
        loss = loss_object( y, model(x, training=True ) )
    gradients = tape.gradient( loss, model.trainable_variables)
    return gradients, loss


def evaluation( x, y, model, loss_object):
    y_pred = model( x, training=False)
    loss = loss_object( y, y_pred)
    return y_pred, loss


## Generators cannot be tf.function s, it significantly slows down training....
def batch_data( x, y, n_batches, shuffle=True):
    """
    Generator/Factory function, yields 'n_batches' batches when called in a for loop
    The last batch is the largest if the number of samples is not integer divisible by 'n_batches'
    (the last batch is at most 'n_batches-1' larger than the other batches)
    Input:
            x               TF tensor or numpy array
                            input data of the ANN
            x               TF tensor or numpy array
                            output data of the ANN
            n_batches       int
                            number of batches to return
        OPTIONAL
            shuffle=True    bool
                            If the data should be shuffled before batching
    Returns/Yields:
            x_batch         TF tensor or numpy array
                            batched input data
            y_batch         TF tensor or numpy array 
                            batched output data
    """
    n_samples = y.shape[0]
    if shuffle:
        permutation = tf.random.shuffle( tf.range( n_samples) )
        x = x[permutation]
        y = y[permutation]
    batchsize = tf.math.floor( n_samples/ n_batches)
    i = -1 # set a value that for n_batches=1 it does return the whole set
    for i in range( n_batches-1):
        yield x[ i*batchsize:(i+1)*batchsize], y[ i*batchsize:(i+1)*batchsize]
    yield x[(i+1)*batchsize:], y[(i+1)*batchsize:]

