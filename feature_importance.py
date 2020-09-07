import numpy as np
#import tensorflow as tf

def shuffle_features( model, inputs, outputs, error_measure='MSE', evaluations=5):
    """
    Determine the impact of each input feature for every output
    This randomly shuffles the investigated feature between the samples
    while keeping the other features constant, then investigates how much
    error got added. 
    Note that the importance should be scaled.
    Note that the data has to be arranged row wise (each row one sample)
    Parameters:
    -----------
    model:      model object
                model which has to have the method 'evaluate()' or 'predict()'
                i.e. prediction = model.evaluate( inputs) 
    inputs:     numpy nd-array
                inputs of the neural network arranged row wise
    outputs:    numpy nd-array
                corresponding outputs for the inputs
    error_measure:  function or string, default 'MSE'
                    error measure given as a function
                    chosen error measure for comparing the predictions
    evaluations:    int, default 5
                    how often the same feature should be shuffled
                    yields a more sensible measure, flattens out some randomness
    Returns:
    --------
    importance:     numpy nd-array
                    importance array, importance.shape == ( n_outputs, n_features)
    """
    try: 
        prediction    = model.predict( inputs)
    except: prediction = model.evaluate( inputs)
    # TODO give the user some feedback when it doesnt work, otherwise its hard to debug
    # TODO basically what should happen is that when the model can not predict with these functions, tell the user whta happened and return
    #except: raise Exception( "Model does not contain 'predict( x)' nor 'evaluate( x)' method, this function will not work!" )
    if not isinstance(error_measure, str):
        error_function = error_measure
    else:
        if error_measure.lower() == 'mse':
            error_function = lambda prediction, outputs: np.mean( (prediction - outputs)**2, axis=0 )
        elif error_measure.lower() == 'mae':
            error_function = lambda prediction, outputs: np.mean( np.abs(prediction - outputs), axis=0 )
        #elif ....
    reference_error = error_function( prediction, outputs)

    n_samples  = inputs.shape[0]
    n_features = inputs.shape[1]
    n_outputs  = outputs.shape[1]
    importance = np.zeros( (n_outputs, n_features) )
    for i_feature in range( n_features ):
        accumulated_error = 0
        shuffled_inputs = inputs.copy()
        for _ in range( evaluations):
            shuffle            = np.random.permutation( n_samples)
            shuffled_inputs[ :, i_feature] = inputs [ shuffle, i_feature]
            try: prediction    = model.predict( shuffled_inputs)
            except: prediction = model.evaluate( shuffled_inputs)
            accumulated_error += error_function( prediction, outputs) / reference_error
        importance[:,i_feature] = accumulated_error / evaluations
    return importance


def weight_inspection( weights, normalization=1): 
    """
    Estimate the importance of parameters based on their weights connecting them to the first hidden layer.
    Sums up all the absolute value of the  weights up to the first layer, then everything is intertwined.
    Only works for a dense neural network or any other model which has weights connecting
    to the first hidden layer with a_next = x @ W + b
    This function is only able to capture the feature's overall importance
    Parameters:
    -----------
    weights:        numpy nd.array
                    weights of the model to the first layer
    normalization:  numpy 1d-array, default 1
                    if the features are not normalized, this vector
                    should normalize the features to scale 
    Returns:
    --------
    importance:     numpy 1d-array
                    overall importance vector of the inputs
                    based on their weights
    """
    n_features = weights.shape[0]
    weights = weights / normalization
    return np.abs( weights).sum( 1)


def feature_variation( model, inputs, outputs, increments=15, constant=1):
    """
    NOTE: This function does not do what i expected, seems like every parameter
    when varying is able to modify the outputs such that the whole interval of
    possible output values is covered

    Vary one of the features through its present range (from min to max) and
    inspect what range of output values the ANN can take on. The other features
    are held constant while varying another.
    This function returns the spectrum of achieved values by varying the 
    parameters. The bigger the spectrum, the more important the feature
    Note that the calculated importance should be scaled
    Parameters:
    -----------
    model:      model object
                model which has to have the method 'evaluate()' or 'predict()'
                i.e. prediction = model.evaluate( inputs) 
    inputs:     numpy nd-array
                inputs of the neural network arranged row wise
    outputs:    numpy nd-array
                corresponding outputs for the inputs
    increments: int, default 15
                in how many increments the spectrum of inputs should be sampled
    constant:   int, default 1
                choose between 0 and 1
                0: all values except the varied one are set to 0
                1: all values except the varied one are kept constant as input
    Returns:
    --------
    importance: numpy nd-array
                importance array, importance.shape == ( n_outputs, n_features)
    """
    n_samples  = inputs.shape[0]
    n_features = inputs.shape[1]
    n_outputs  = outputs.shape[1]
    minimum_prediction = 1e10 * np.ones( n_outputs)
    maximum_prediction = -minimum_prediction
    importance = np.zeros( ( n_outputs, n_features ))

    for i in range( n_features):
        lower_bound = np.min( inputs[:, i] )  /2
        interval = (np.max( inputs[:, i] ) - lower_bound )/2
        if constant is 0:
            x = np.zeros( ( 1, n_features )) 
        else:
            x = inputs.copy()

        for step in range( increments +1 ):
            current_value = lower_bound + step/increments * interval
            drawn_sample = np.random.randint( n_samples)
            x[ drawn_sample, i] = current_value 
            try: prediction    = model.predict( x)
            except: prediction = model.evaluate( x) 
            maximum_prediction = np.maximum( np.max( prediction, axis=0), maximum_prediction) 
            minimum_prediction = np.minimum( np.min( prediction, axis=0), minimum_prediction) 
            print( 'my current max and min of the prediction', np.max( prediction, axis=0), np.min( prediction, axis=0) )
        importance[:, i] = maximum_prediction - minimum_prediction
    return importance 
