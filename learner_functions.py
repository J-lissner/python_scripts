import tensorflow as tf
import numpy as np 


#@tf.function decorator not written here, because it does not allow to train multiple ANN
def train_step(x, y, model, loss_object, **model_kwargs):
    """
    Compute the gradients of the loss w.r.t. the trainable variables of the model
    Parameters:
    -----------
    x:              tensorflow.tensor like
                    training input data aranged row wise (each row 1 sample)
    x:              tensorflow.tensor like
                    training output data aranged row wise 
    model:          tf.keras.Model object
                    model which predicts the training data
    loss_object:    tf.keras.losses (or self defined)
                    loss object/function to evaluate the loss
    **model_kwargs: keyworded arguments, no defaults
                    additional kwargs for the model call
    """
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True, **model_kwargs )
        loss = loss_object( y, y_pred )
    gradients = tape.gradient( loss, model.trainable_variables)
    return gradients, loss, y_pred


def relative_mse( y, y_pred, axis=None):
    """
    Compute the relative MSE for the predicted values. The mse 
    is analogue to the frobenius norm for default inputs (and real valued y)
    if axis is specified it can also compute it component wise.
    Note that tf_function decorator makes it unable to store in pickle,
    but might help efficiency
    Parameters:
    -----------
    y:      tensorflow tensor
            target value
    y_pred: tensorflow tensor
            predicted_value
    axis:   bool, default None
            which axis to reduce, reduces to scalar value per default 
    Returns:
    --------
    loss:   scalar or tensorflow tensor
            loss of the prediction
    """
    y_norm = tf.reduce_sum( tf.square(y), axis=axis )
    error = tf.reduce_sum( tf.square(y-y_pred), axis=axis)
    loss = error/y_norm
    return loss**0.5


class LinearSchedule( tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__( self, max_learnrate=0.05, min_learnrate=5e-3, decay_epoch=25, static_epoch=150):
        """
        Have a scheduled leraning rate which is like this ~~~\_____
        Constant at first, linearly decaying and then constant again.
        Parameters:
        -----------
        max_learnate:   float, default 0.1
                        learning rate at the start
        min_learnate:   float, default 0.005
                        learning rate at the 'end' to 'inf'
        decay_epochs:   int, default 25
                        after how many epochs the rate should declines
        static_epochs:  int, default 150
                        after how many epochs the minimum is reached
        """
        self.start = max_learnrate
        self.end = min_learnrate
        self.decay_epoch = decay_epoch
        self.static_epoch = static_epoch

    def __call__( self, step):
        if step >= self.static_epoch: #most common case
            return self.end
        elif step < self.decay_epoch:
            return self.start
        else: 
            slope = (step - self.decay_epoch)/ (self.static_epoch - self.decay_epoch )
            return self.start - slope * (self.start - self.end)





### deprecrated or tests

def train(epochs, stopping_delay, x_train, y_train, n_batches, model, optimizer, loss_metric, x_valid, y_valid):
    """
    # This was just one idea, but i think i will scratch that
    def train( model, n_epoch, stopping_delay, spike_delay, n_batches, *data, **minimizers):
    bruh idk if i should have a function with that many parameters
    I think if i just put all the optimizer and loss stuff to kwargs then its managable, also *data, **minimizers
    """
    while i <= epochs and decline <= stopping_delay: 
        batched_data = get.batch_data( x_train, y_train, n_batches )
        batch_loss = []
        k = 0
        for x_batch, y_batch in batched_data:
            gradient, _, y_pred = train_step( x_batch, y_batch, ANN, cost_function) 
            optimizer.apply_gradients( zip(gradient, ANN.trainable_variables) )
            k += 1
            batch_loss.append( loss_metric( y_batch, y_pred) )
        loss_t = np.mean( batch_loss)

        # epoch post processing
        _, loss_v = evaluate( x_valid, y_valid, ANN, loss_metric)
        valid_loss.append( loss_v.numpy() )
        train_loss.append( loss_t ) 
        if valid_loss[-1] < valid_loss[best_epoch]: #TODO more sophisticated criteria
            #current idea: also that the validation is not e.g. 10 times larger than the training error
            best_epoch = i
            decline    = 0
            checkpoint_manager.save() 
        print( 'this is my spike counter{} this is my got_worse {}'.format( spike_counter, decline) )
        if valid_loss[-1] < train_loss[-1]: #while underfit, keep training
            spike_counter += 1
            if spike_counter >= spike_delay:
                print( '!!!!!!!!!!!!!!!! i am resetting my decline because i am underfit !!!!!!!!!!!!!!!!!!1')
                decline = 0
        else:
            spike_counter = 0
        decline += 1
        i += 1
        if i % 500 == 0:
            toc('trained for 500 epochs')
            print( 'current validation loss:\t{:.5f} vs best:\t{:.5f}'.format( valid_loss[-1], valid_loss[best_epoch] ) )
            print( 'vs current train loss:  \t{:.5f} vs best:\t{:.5f}'.format( train_loss[-1], train_loss[best_epoch] ) )
            tic('trained for 500 epochs', True)
    return model, losses

