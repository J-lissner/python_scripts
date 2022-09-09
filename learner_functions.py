import tensorflow as tf
import numpy as np 
from numpy import cos, pi, sin


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
        loss   = loss_object( y, y_pred )
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
    error  = tf.reduce_sum( tf.square(y-y_pred), axis=axis)
    loss   = error/y_norm
    return loss**0.5

def slashable_lr():
    return (JumpingLR, SuperConvergence, RemoteLR )

thetas = []
def estimate_max_lr( model, learnrate):
    """
    call this every iteration, i.e. batch and get an estimate of the LR
    pass the model and the current learnrate
    """
    if len( thetas) <3:
        thetas.append( [x.numpy().copy() for x in model.trainable_variables] )
    elif len( thetas) == 3:
      upper_part = sum([ np.abs( x - y).sum() for x,y in zip( thetas[1], thetas[0] )] )
      lower_part = sum([ np.abs( 2*x - y - z ).sum() for x,y,z in zip( thetas[1], thetas[0], thetas[2] )] )
      print( 'estimated maxumim learnrate:', learnrate*upper_part/lower_part )
      for i in range( 3):
        thetas[i] = None
      thetas.append(None)

class LRSchedules(  tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    This is the parent class for all my variable learning rates to inherit 
    some default behaviour and methods
    """
    def __init__( self, *args, **kwargs):
        """
        Set some generic default learnrate to be callable below
        """
        self.learnrate = 1e-3

    ## methods to reference internal variables for other methods
    def reference_optimizer( self, optimizer):
        """ 
        Reference the optimizer which (should be) resettable during slash
        """
        self.optimizer       = optimizer
        self.optimizer_state = optimizer.variables().copy()

    def reference_model( self, model):
        """ 
        Reference the model to compute the max LR
        """
        self.model = model

    def reset_optimizer(self, optimizer=None, state=None):
        """
        reset the optimizer to initial state for every variable in it
        besides the learning rate if it is a variable one.
        Can be called as function with any object as placeholder for self
        Parameters:
        -----------
        optimizer:      optimizer objects with 'variables()', default None
                        tensorflow otpimizer, defaults to the optimizer
                        from allocation of this object
        state:          optimizer.variables()
                        State to which the optimizer should revert, defaults
                        to the state on this object invocation
        Returns:
        --------
        None:           does all operations in place
        """
        optimizer = self.optimizer if optimizer is None else optimizer
        if optimizer is None:
            return
        state         = self.optimizer_state if state is None else state
        current_state = optimizer.variables() 
        for var, i in zip( state, range( len( state) )):
            if isinstance( state[i], slashable_lr()):
                continue
            current_state[i] = var 

    def estimate_max_lr( self):
        """
        Estimate the maximum admissible learning rate based off the models
        parameters consecutive three steps. Equation taken from
        https://doi.org/10.1117/12.2520589 (8) on page 5
        Must have the model referenced for this to work
        Must be called three consecutive times in the loop
        """
        if hasattr( self, 'max_lr' ):
            return self.max_lr
        if not hasattr( self, 'model_parameters' ):
            self.model_parameters = []
        self.model_parameters.append( [x.numpy().copy() for x in self.model.trainable_variables] )
        if len( self.model_parameters) == 3:
          params         = self.model_parameters
          upper_part     = sum([ np.abs( x - y).sum() for x,y in zip( params[1], params[0] )] )
          lower_part     = sum([ np.abs( 2*x - y - z ).sum() for x,y,z in zip( params[1], params[0], params[2] )] )
          self.max_lr    = self.learnrate * upper_part/lower_part
          print( f'The maximum estimated learning rate is: {self.learnrate}' )
          del self.model_parameters, params, self.model
          return self.max_lr

    def slash( self, slash=10**0.5):
        self.learnrate /= slash 

    def __call__( self, step):
        return self.learnrate


class RemoteLR(LRSchedules):
    """ this is a LR which will be affected mostly by remote operations 
    the max learning rate will be estimated during runtime learning and 
    adjusted accordingly
    first it will be constant until plateau, then there will be 2 up jumps
    with a fixed amount of steps and thereafter downsteps whenever it 
    plateaus"""
    def __init__( self, n_steps=50, base_lr=1e-3, jump=10**0.5, jump_up=True, n_down=3 ):
        """
        Parameters:
        -----------
        n_steps:        int, default 100
                        how many steps the learnrate should be forced
                        constant when we are upscaling the learnrate
        base_lr:        float, default 1e-3
                        learnrate to start out with, max elligible leranrate will
                        be estimated during runtime, and base_lr adjusted to 
                        max( base_lr, max_lr/jump )
        jump_up:        bool, default True
                        if the learning rate should jump up twice to max_lr
        jump:           float, default 10**0.5
                        how big each jump should be
        n_down:         int, default 3
                        how often the learning rate should be decreased by jump
                        If jump_up is specified then the number of up jumps will
                        be added to n_down
        """
        self.learnrate = base_lr
        self.jump      = jump
        self.n_steps   = n_steps
        self.n_up      = 2 if jump_up else 0
        self.n_down    = n_down + self.n_up 
        # other variables required later during evaluateion
        self.phase            = 0 #phases are: constant, (jump up to max), slash down by jump
        self.optimizer        = None
        self.model            = None
        self.max_lr           = None
        self.allow_stopping   = False 
        self.model_parameters = []
        self.first_slash      = 1/99 #any float value to false the ==


    def slash( self, remote_call=True):
        """
        Adjust the learning rate, only allow for an interference from
        a remote call when not increasing the leranrate
        """
        ## if we are at the first phase of constant lr
        if self.phase == 0:
            self.phase += 1
            self.reset_optimizer()
            if self.n_up != 0:
                self.learnrate *= self.jump 
                print( 'increasing learning rate')
            else:
                self.learnrate /= self.jump 
                print( 'decreasing learning rate' )
        ## if we are currently in the up path
        elif 0 < self.phase < self.n_up and not remote_call: #fixed duration
            print( 'increasing learning date')
            self.reset_optimizer()
            self.learnrate *= self.jump
            self.phase += 1
        ## if we are slashing it from outside upon plateau
        elif self.n_up <= self.phase <= (self.n_up + self.n_down):
            print( 'decreasing learning rate')
            self.phase += 1
            self.reset_optimizer()
            self.learnrate /= self.jump
            if self.phase == (self.n_up + self.n_down):
                print( 'now enabling early stopping switch' )
                self.allow_stopping = True #finally enable stopping of the training


    def __call__(self, step):
        """
        Return the current learnrate. Also estimates the max LR in the first few steps
        """
        ## estimate the maximum learning rate
        if self.max_lr is None and self.model is not None:
            self.model_parameters.append( [x.numpy().copy() for x in self.model.trainable_variables] )
            if len( self.model_parameters) == 3:
              params         = self.model_parameters
              upper_part     = sum([ np.abs( x - y).sum() for x,y in zip( params[1], params[0] )] )
              lower_part     = sum([ np.abs( 2*x - y - z ).sum() for x,y,z in zip( params[1], params[0], params[2] )] )
              self.max_lr    = self.learnrate * upper_part/lower_part
              self.learnrate = max( self.learnrate, self.max_lr/self.jump )
              print( f'adjusting learning rate to be {self.learnrate}, maximum found={self.max_lr}' )
              del self.model_parameters, params
        ### internally adjust the learning rate when increasing it
        if 0 < self.phase <= self.n_up and (step - self.first_slash) % self.n_steps == 0:
            self.slash( remote_call=False)
        if self.phase == 1 and isinstance( self.first_slash, float): #slashed from outside on first plateau
            self.first_slash = step #tracking parameter 
        return self.learnrate 






class SuperConvergence(LRSchedules):
    """
    Implementation of a 1cycle learning rate cycler, after the first cycle
    the learning rate is held constant with slashing after <slash_delay> epochs,
    or can be triggered from outside.
    The rise and slope of learning rate is implemented with a cosine 
    """
    def __init__( self, n_batches, max_learnrate, learnrate=1e-3, n_slash=2, slash_delay=400, n_epochs=8, warmup=30, **functions ):
        """
        Initialize counters and functions
        The functions are the following
              /\ up and down in cos, then slashing every slash_delay
        -----/  \--...________
        MB TO BE IMPLEMENTED: warmup with constant min_learnrate
        Parameters:
        -----------
        n_batches:      int
                        how many batches per iteration
        max_learnrate:  float
                        What the maximum leraning rate should be
        learnrate:      float, default 2e-2
                        base learnrate and after learnrate phase 2
        n_slash:        int, default 3
                        how often to divide the learnrate by 10 before allowing
                        the early stop to go off
        slash_delay:    int, default 400
                        after how many steps - not epochs - the learnrate 
                        should be slashed if not triggered beforehand
        n_epochs:       int, default 8
                        how many epochs before concluding 1 phase
        warmup:         int, default 10
                        how many epochs before the learning ratei s increased
        """
        self.warmup           = warmup * n_batches
        self.steps            = n_epochs * n_batches
        self.n_slash          = n_slash
        self.slash_delay      = slash_delay 
        self.holdout_counter  = 0
        self.slash_factor     = 10
        self.previous_slashes = 0
        self.max_learnrate    = max_learnrate
        self.learnrate        = learnrate
        phase_difference      = (max_learnrate - self.learnrate) /2 #divided by factor to because cos is shifted
        self.up_function      = lambda x: self.learnrate + phase_difference * (-cos( pi*x/self.steps ) + 1)
        self.down_function    = lambda x: max_learnrate + phase_difference * (cos( pi*(x-self.steps)/self.steps ) -1 )
        self.allow_stopping   = False

    def slash( self):
        """ slash the current learning rate by <slash_factor>, reinitialize counters """
        self.learnrate /= self.slash_factor
        self.holdout_counter    = 0
        self.previous_slashes += 1
        if self.previous_slashes > self.n_slash:
                self.allow_stopping = True

    def __call__( self, step):
        if step > 2*self.steps + self.warmup:
            if self.holdout_counter >= self.slash_delay:
                self.slash()
            self.holdout_counter += 1
            return self.learnrate 
            ## last phase with constant learnrate and slashes, most common occurence
            pass
        elif step > 1*self.steps + self.warmup: #second phase of cooldown
            return self.down_function( step)
        elif step > self.warmup:  #first phase of increasing
            return self.up_function( step)
        else:
            return self.learnrate  #warmup

class JumpingLR( tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Have a constant but jumpy learning rate
    Starts of with the base learnrate, increases it n-amount of times, 
    and then decreases again
    """
    def __init__( self, n_batches, base_learnrate=1e-3, scaling_factor=10**0.5, **timers):
        """
        Initialize internal variables, does contain default keyworded
        arguments
        Parameters:
        -----------
        n_batches:      int,
                        batches per epoch
        base_learnrate: float, default 1e-3
                        learnrate with which the algorithm starts out
        scaling_factor: float, default \sqrt(10)
                        factor of which to adjust the learning rate each jump
        **timers with default kwargs
          n_up:         int, default 2
                        how many jumps upward (these come first)
          n_down:       int, default 6
                        how many jumps downward (these come later
          holdout_delay:int, default 10
                        how long to wait at most between each lr jump
          warmup:       int, default 20
                        how many epochs to have before the first phase 
        """
        self.n_up          = timers.pop( 'n_up', 2 ) +1
        self.n_down        = timers.pop( 'n_down', 6 )
        self.holdout_delay = timers.pop( 'holdout_delay', 10 ) * n_batches
        self.warmup        = timers.pop( 'warmup', 20  ) *n_batches
        if self.warmup == 0: 
            self.warmup = 1 #idk if calling starts at 0 or 1, makes sure
        self.learnrate      = base_learnrate
        self.steps          = n_batches * self.holdout_delay
        self.phase          = 0 if self.warmup >= 1 else 1
        self.scaling_factor = scaling_factor
        self.allow_stopping = False


    def readjust( self ):
        """ 
        Adjust the learning rate by either jumping up or down,
        depending on the amount of previous calls
        """
        if 0 < self.phase < self.n_up:
            self.learnrate *= self.scaling_factor 
        elif self.phase < self.n_down + self.n_up:
            self.learnrate /= self.scaling_factor 
        self.phase += 1 #which phas of learning rate
        if self.phase == self.n_down + self.n_up:
            self.allow_stopping = True

    def slash( self, *args, **kwargs):
        """ shadow readjust to have method compliance"""
        return self.readjust( *args, **kwargs)

    def __call__( self, step):
        if step == self.warmup:
            self.readjust()
        elif (step - self.warmup) % self.holdout_delay == 0:
            self.readjust() 
        return self.learnrate




class LinearSchedule(LRSchedules):
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
        self.start        = max_learnrate
        self.end          = min_learnrate
        self.decay_epoch  = decay_epoch
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
        batch_loss   = []
        k            = 0
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

