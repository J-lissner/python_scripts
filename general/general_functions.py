import numpy as np
from timers import tic, toc #just that they are present in namespace and can be imported here (kinda dirty but w/e)

class Cycler():
    """
    define a cycling iterator of an iterator which returns 2 values
    upon next() invokation. Function is intended to be used for
    torch.utils.data.DataLoader for an endless cycling
    This class is only meant to be used with <next( Cycler)>
    """
    def __init__( self, iterator, iteration_type=iter):
        """
        Allocate the iterating object as well as a copy of the <iterator>
        Doubles the memory requirement for the <iterator> object!
        Parameters:
        -----------
        iterator:           python object
                            python object to iterate with
        iteration_type:     function, default iter
                            type of iterator to store, governs the return of
                            next(). Use e.g. iter or enumerate
        """
        self.original_iterator = iterator
        self.iterator = iteration_type( iterator)
        self.iteration_type = iteration_type

    def __next__( self):
        try:
            return next( self.iterator)
        except:
            self.iterator = self.iteration_type( self.original_iterator)
            return next( self.iterator)

    def __getitem__(self, idx):
        return self.original_iterator[idx]

    def __iter__(self):
        return iter( self.original_iterator )

    def __len__(self):
        return len( self.original_iterator)


def print_functions( module, display_format=', '):
    """
    A function to print out all the functions/submodules contained within the given module
    Parameters:
    -----------
    module:         imported module
                    variable name pointing to the module which should be inspected
    display_format: string, default ', '
                    format to put after each found function in the module
                    advised strings are '\n' or ', ' (default)
    Retuns:
    -------
    None:       prints the functions to consolue
    """
    try:
        print( 'checking module/object {}'.format( module.__name__), '\n' )
    except:
        print( 'checking the given module/object')
    found_functions = []
    [ found_functions.append( x) for x in dir( module) if not '_' in x[0] ]
    print_string = ''
    for function in found_functions:
        print_string = print_string + function + display_format
    print( print_string)
    return


def periodic_insertion( insert_array, read_array, bounds ):
    """
    Periodically put "read_array" into "full array"
    "read_array" contains negative and positive "bounds" denoting
    the slices which assemble "read_array" into "insert_array"
    It is assumed that the operation is possible/the dimensions match
    The function also only deals with positive increment slicing!
    Parameters:
    -----------
    full_array:     numpy nd-array
                    array in which read_array is inserted
    read_array:     numpy nd-array
                    array which is inserted according to bounds
    bounds:         numpy 2d-array
                    bounds of the slice organized column wise
                    (each column denotes one index)
    Returns:
    --------
    full_array:     numpy nd-array
                    array with read array periodically inserted
    """
    ## Input preprocessing
    n_idx = bounds.shape[1] 
    bounds = bounds % np.array( read_array.shape) 
    split_dim = [] #first check if Section needs to be split up, and then in how many parts
    for i in range( n_idx):
        if bounds[0,i ] > bounds[1,i]:
            split_dim.append( i ) 
    read_voxels = read_array.shape
    # initialize the sliced Insert-position and the sliced Section for the furhter loop
    if 0 not in split_dim: 
        slice_pos    = [ [slice( bounds[0,0], bounds[1,0] )] ]
        slice_chunks = [ [slice( None, None, -1 )] ]
    else:
        slice_pos    = [ [slice( bounds[0,0], None) ], [ slice( None, bounds[1,0] ) ] ]
        slice_chunks = [ [slice( None, bounds[0,0]-read_voxels[0] -1, -1) ], [ slice( bounds[0,0]-read_voxels[0]-1, None, -1 ) ] ]

    ### looping for insertion positions
    # go through the remaining dimensions and compute the chunking
    for i in range( 1, n_idx):
        if i not in split_dim:
            for split_slice in slice_pos:
                split_slice.append( slice( bounds[0,i], bounds[1,i] ) )
            for split_chunk in slice_chunks:
                split_chunk.append( slice( None, None, -1 ) )
        # Nest (Branch out) the slices for each dimension it needs to be split in
        elif i in split_dim:
            new_pos = []
            new_chunks = []
            for j in range( len(slice_pos) ):
                for splits in [ slice( bounds[0,i], None ), slice( None, bounds[1,i] ) ]:
                    new_pos.append( slice_pos[j].copy() + [splits] )
                for chunks in [ slice( None, bounds[0,i]-read_voxels[i]-1, -1), slice( bounds[0,i]-read_voxels[i]-1, None, -1 ) ]:
                    new_chunks.append( slice_chunks[j].copy() + [chunks] )
            slice_pos = new_pos.copy()
            slice_chunks = new_chunks.copy() 
    ## insert the subimage inside the full image
    slice_pos = [ tuple(split_slice) for split_slice in slice_pos]
    slice_chunks = [ tuple(split_chunk) for split_chunk in slice_chunks]
    for i in range( len(slice_pos) ):
        insert_array[ slice_chunks[i] ] += read_array[ slice_pos[i]]
    return insert_array



def bin_indices( x, x_binned=None, n_bins=15 ):
    """
    Find the indices in <x> which are inside each bin defined in x_binned.
    If <x_binned> is not specified, <x> is binned into <n_bins> uniform
    bins.  
    This function might not work for descending x_binned, has not been tested.
    Parameters:
    -----------
    x:              numpy 1d-array
                    given data
    x_binned:       numpy 1d-array
                    bin bounds to which the data should be sorted
    n_bins:         int, default 15
                    How many bins the data should be put into
                    if x_binned is not specified 
    Returns:
    --------
    bin_indices:    list of numpy 1d-arrays
                    indices for each bin, has of length len( x_binned)
    x_binned:       OPTIONAL, numpy 1d-array
                    bin intervals if x_binned was not given as input 
    """
    if x_binned is None:
        given_bins = False
        bin_incr = (x.max() - x.min() )/ (n_bins-1)
        x_binned = np.arange( x.min(), x.max() + 0.5*bin_incr, bin_incr)
        assert len( x_binned) == n_bins, 'got more bins than requested, got {}, wanted {}'.format( len( x_binned), n_bins) 
    else:
        given_bins = True
        n_bins = len( x_binned)

    #new approach rewritten by me
    x_idx = []
    for i in range( n_bins-1):
        x_idx.append( np.argwhere( (x_binned[i] < x) * (x <= x_binned[i+1]) ) )
    x_idx.append( np.argwhere( x_binned[-2] < x ) )
    if given_bins:
        return x_idx
    else:
        return x_idx, x_binned




def bin_data( data, n_bins=None ):
    """
    Count the data and return the number of samples per bin
    If the number if n_bins is not specified it will automatically be computed
    Parameters:
    -----------
    data:       numpy 1d-array
                given data
    n_bins:     int, default None
                How many bins the data should be put into
                Computes a sensitive default value if not specified

    Returns:
    --------
    count:      numpy nd-array
                number of samples in the given bin
    center:     numpy nd-array
                center value of the given bin
    width:      float
                width of the bin
                bin start/end is center -/+ width/2
    """
    n_samples = data.shape[0]

    # automatical choice for the number of bins
    if n_bins is None:
        if n_samples < 100:
            print( 'too few data samples present, returning un-binned data' )
            return [data.copy()]
        else:
            # Square-root choice
            n_bins = int(np.ceil(np.sqrt(n_sample)))
            
    inspected_values = data[:].flatten()
    sorting          = np.argsort( inspected_values)
    data             = data[ sorting ] 
    data_bins   = []
    lower_bound = inspected_values.min()
    upper_bound = inspected_values.max() - lower_bound
    stepsize    = upper_bound/ n_bins

    center      = (0.5 + np.arange(n_bins))*stepsize+lower_bound
    width       = stepsize
    count       = np.zeros(n_bins)
    previous_sample = 0
    for i_bin in range( 0,n_bins-1):
        for j in range( previous_sample, n_samples):
            if data[ j ] > center[i_bin]+0.5*stepsize:
                count[i_bin] = j-previous_sample
                previous_sample = j 
                break
    count[-1] = n_samples-count[0:-1].sum()
    return count, center, width
