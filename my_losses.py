import tensorflow as tf

def shifted_neg_log_likelihood(y, rv_y, shift=-0.25):
    """
    Shifted neg-log-likelihood function along the X-axis
    to have a not as steep penalization on probabilities close to 0. 
    Parameters:
    -----------
    y           : true value
    rv_y        : random variable (distribution)
    shift:      float, default 0.25
                x position of -inf of the log value
    Returns:
    --------
    neg_log_lik : scalr left-shifted neg-log-likelihood  
    """
    prob      = rv_y.prob(y) # probability value for y being a part of rv_y
    log_prob  = tf.math.log(0.5*(prob-shift)) # log of the shifted probability
    neg_log_lik = -tf.reduce_mean(log_prob) # convert array to scalar
    return neg_log_lik

