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


### different loss functions
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


##### Learning rate schedule related things
def slashable_lr():
    """ have all lr objects  which contain the slash function (must inherit
    from father method) in one tuple, used for isinstance comparison"""
    scheduler = LRSchedules
    return tuple( [scheduler] + scheduler().children() )

def is_slashable( lr_object):
    """ Directly return true or false if you can slahs your learning rate"""
    custom_lrs = slashable_lr()
    return isinstance( lr_object, custom_lrs) or (lr_object in custom_lrs )


class LRSchedules( tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    This is the parent class for all my variable learning rates to inherit 
    some default behaviour and methods
    """
    def __init__( self, *args, **kwargs):
        """
        Set some generic default learnrate to be callable below
        """
        self.learnrate = 1e-3
        self.phase = 0
        self.allow_convergence = True

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
        if hasattr( self, 'max_lr' ) and self.max_lr is not None:
            return self.max_lr
        if not hasattr( self, 'model_parameters' ):
            self.model_parameters = []
        self.model_parameters.append( [x.numpy().copy() for x in self.model.trainable_variables] )
        if len( self.model_parameters) == 3:
          params         = self.model_parameters
          upper_part     = sum([ np.abs( x - y).sum() for x,y in zip( params[1], params[0] )] )
          lower_part     = sum([ np.abs( 2*x - y - z ).sum() for x,y,z in zip( params[1], params[0], params[2] )] )
          self.max_lr    = self.learnrate * upper_part/lower_part
          self.learnrate = max( self.learnrate, self.max_lr/self.jump )
          print( f'adjusting learning rate={self.learnrate:1.3e}, maximum found={self.max_lr:1.3e}' )
          del self.model_parameters, params, self.model
          return self.max_lr

    def slash( self, slash=10**0.5):
        """ adjust the learning rate from a remote trigger"""
        self.learnrate /= slash 

    def __call__( self, step):
        return self.learnrate

    @classmethod
    def children( self):
        return self.__subclasses__()


class RemoteLR(LRSchedules):
    """ this is a LR which will be affected mostly by remote operations 
    the max learning rate will be estimated during runtime learning and 
    adjusted accordingly
    first it will be constant until plateau, then there will be 1 up jumps
    with a fixed amount of steps and thereafter downsteps whenever it 
    plateaus"""
    def __init__( self, base_lr=1e-3, n_steps=50, jump=5, n_up=1, n_down=3, slash_decay=3 ):
        """
        Note that the results were pretty dependent on the weight decay
        from adamW, it performed significantly more reliable for a decay
        of 1e-4
        Parameters:
        -----------
        base_lr:        float, default 1e-3
                        learnrate to start out with, max elligible leranrate will
                        be estimated during runtime, and base_lr adjusted to 
                        max( base_lr, max_lr/jump )
        n_steps:        int, default 50
                        how many steps the learnrate should be forced
                        constant when we are upscaling the learnrate
        n_up:           int, default 0
                        How many times the learning rate should jump up, 
                        If the number is larger than 1 it will exceed 'max_lr'
        jump:           float, default 5
                        how big each jump should be
        n_down:         int, default 3
                        how often the learning rate should be decreased by jump
                        If n_up>0 then the number of up jumps will be added to n_down
        slash_decay:    float, default 3
                        slashes the weight decay upon LR decrease, even after the jumps
                        up. Set to False if it should not be slashed
        """
        self.learnrate = base_lr
        self.jump      = jump
        self.n_steps   = n_steps
        self.n_up      = n_up
        self.n_down    = n_down + self.n_up 
        # other variables required later during evaluateion
        self.phase            = 0 #phases are: constant, (jump up to max), slash down by jump
        self.optimizer        = None
        self.model            = None
        self.max_lr           = None
        self.allow_stopping   = False 
        self.model_parameters = []
        self.first_slash      = 1/99 #any float value to false the == condition
        self.slash_decay = slash_decay #whether or not to slash the weight decay
        self.slash_epoch = []
        self.step = 0


    def slash( self, remote_call=True):
        """
        Adjust the learning rate, only allow for an interference from
        a remote call when not increasing the leranrate
        """
        self.slash_epoch.append( self.step) #tracking variable
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
            print( 'decreasing learning rate', end='')
            self.phase += 1
            self.reset_optimizer()
            self.learnrate /= self.jump
            if self.slash_decay:
              try: 
                self.optimizer.weight_decay = self.optimizer.weight_decay / self.slash_decay
                print( ' and weight decay' )
              except: 
                print( ', failed to decrease weight decay' ) 
            else: print()
            if self.phase == (self.n_up + self.n_down):
                print( 'now enabling early stopping switch' )
                self.allow_stopping = True #finally enable stopping of the training


    def __call__(self, step):
        """
        Return the current learnrate. Also estimates the max LR in the first few steps
        """
        ## estimate the maximum learning rate
        self.step += 1
        if self.max_lr is None and self.model is not None:
            self.estimate_max_lr()
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




