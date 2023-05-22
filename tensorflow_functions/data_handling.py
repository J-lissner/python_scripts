import numpy as np
try:
    import tensorflow as tf
    import tensorflow_addons as tfa
    import tensorflow_functions as tfun
except:
    print( 'unable to load tensorflow related modules, some functions in "data_handling" will be unavailable!' )
import random
from math import ceil, floor

def batch_data( n_batches, data, shuffle=True):
    """
    Generator/Factory function, yields 'n_batches' batches when called in a for loop
    The last batch is the largest if the number of samples is not integer divisible by 'n_batches'
    (the last batch is at most 'n_batches-1' larger than the other batches)
    things as i want
    Parameters
    ----------
    n_batches       int
                    number of batches to return
    data:           list of tensorflow.tensors
                    Tensorflow tensors which should be batched
    shuffle         bool, default True
                    If the data should be shuffled during batching
    Yields:
    -------
    batches:        tuple of tensorflow tensors
                    all of the tensors batched as in given order 
    """
    n_samples = data[-1].shape[0]
    if shuffle:
        permutation = tf.random.shuffle( tf.range( n_samples, dtype=tf.int32) )
    else:
        permutation = tf.range( n_samples, dtype=tf.int32 ) 
    batchsize = int( tf.math.floor( n_samples/ n_batches) )
    i         = -1 # set a value that for n_batches=1 it does return the whole set
    for i in range( n_batches-1):
        idx   = permutation[i*batchsize:(i+1)*batchsize]
        batch = []
        for x in data:
            batch.append( tf.gather( x, idx) )
        yield batch
    idx   = permutation[(i+1)*batchsize:]
    batch = []
    for x in data:
        batch.append( tf.gather( x, idx) )
    yield batch


class MultiSetBatcher():
    def __init__( self, data_sets, batch_size=25, shuffle=True ):
        """
        Given multiple sets of non-mergable tenors, return them batchwise
        in matching input output forms. The class only assigns references
        to the dataset objects, does not double memory usage.
        Intended use: For training of resolution independent Conv Nets 
        where the Conv Net sees different resolutions during the training.  
        Called as an iterator, i.e. 
        for x_batch, y_batch  in MultiSetBatcher(): #after invocation
        Parameters:
        -----------
        data_sets:  list like of [[input, ..., output],...] tensor lists
                    data 'tuples' of non mergable tensors
        batch_size: int or list of ints, default 25
                    size of each batch for all or for each training set
        shuffle:    bool, default True
        """
        self.data       = data_sets
        self.batch_size = batch_size
        self.n_datasets = len( data_sets)
        if isinstance( batch_size, int):
            batch_size = self.n_datasets*[batch_size]
        n_samples      = self.__len__()
        self.n_batches = [ ceil(n_samples[i]/batch_size[i]) for i in range( self.n_datasets) ]

    def __len__( self):
        """ 
        Returns the number of samples in each data pair
        """ 
        return [ x[0].shape[0] for x in self.data]

    def __iter__( self):
        """
        Yield the batches iteratively in a memory friendly manner.
        Does go through the entirety of the data when called and yields
        the batches of the different sets (i.e. different resolution) in
        completely random order until each set is exhausted.
        """
        idx               = self.n_datasets*[0] #current idx for batch start/stop
        remaining_batches = self.n_batches.copy()
        while sum( remaining_batches) > 0:
            if sum( remaining_batches) <= self.n_datasets: 
                dset_idx = []
                for i in range( len( remaining_batches)):
                    dset_idx += remaining_batches[i]*[i]
                random.shuffle( dset_idx)
            else: #loop through as many batches as you have datasets
                dset_idx  = self.weighted_sampling( remaining_batches, self.n_datasets) 
            drawn_idx = np.zeros( len(remaining_batches) )
            for i in dset_idx: #backcheck if its not too many for single set
                drawn_idx[i] += 1
            if drawn_idx > remaining_batches.any(): 
                continue #simply draw another one, this will almost never happen 
            for i in dset_idx:
                batch   = []
                ii      = self.batch_size*idx[i]
                jj      = self.batch_size*(idx[i]+1) 
                idx[i] += 1
                remaining_batches[i] -= 1
                for data in self.data[i]: 
                    if remaining_batches[i] != 0:
                        batch.append( data[ii:jj] )
                    else: #last batch might be smaller
                        batch.append( data[ii:] )
                yield batch


    def weighted_sampling( self, sample_weights, n_draws):
        """
        Randomly sample points from a uniform distribution given the 
        <sample_weights>. Draws the index the current pick should fall
        into, i.e. return <n_draws> indices where 
        weight[idx] < pick <= weight[idx+1]
        with 'pick' being uniformly distributed.
        Is a generally usable function and does not have to be bound to the object
        Parameters:
        -----------
        sample_weights: list like of floats > 0
                        weights for each outcome
        n_draws:        int
                        how many draws should be taken
        """
        choices      = n_draws*[None]
        draw_weights = np.cumsum( sample_weights) 
        for i in range( n_draws):
            pick = np.random.uniform(0, sum( sample_weights ))
            j    = 1
            if pick <= draw_weights[0] and draw_weights[0] != 0: #if its smaller than first idx
                choices[i] = 0
            while choices[i] is None:
                if draw_weights[j-1] < pick <= draw_weights[j]:
                    choices[i] = j
                j += 1
        return choices 


    def roll_images( self, roll=True):
        """
        Assuming that all data-pair tuples are variables and images they 
        will be rolled randomly rolled, each set independently
        Calls the 'tf_functions.roll_images' function with default arguments
        Parameters:
        -----------
        roll:       list of ints, default True
                    If every datablock fonud in each datapair should be
                    rolled as well. Otherwise rolls the data at indices i
                    in 'roll'
        """
        for data in self.data:
            if roll is True:
                tfun.roll_images( data)
            else:
                for i in roll:
                    tfun.roll_images( data[i] )


