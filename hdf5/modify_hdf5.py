import h5py
import numpy as np

def add_L( dset_nr, filename, internal_path, start_nr=0, L=np.ones(3), dset_names=None ):
    """
    add a dataset into the given hdf5 file of the same name as the dataset
    with a _L suffix, which contains a float array of [1,1,1] (default)
    In a general case it is assumed that multiple datasets with the name "dset_i"
    are stored in one subfolder of the hdf5 file
    For each dataset, there will be the additional dset_i_L dataset
    Keeps any previously existing _L datasets which already exist, does not create new ones
    Parameters:
    -----------
    dset_nr:        int
                    end counter of the datasets to append the dataset on 
                    (end INCLUSIVE (unlike python))
    filename:       string
                    path and name to the the hdf5 file
    internal_path:  string
                    internal hdf5 path to the folder containing the datasets

    start_nr:       int, DEFAULT 0
                    start counter of the "dset_i"
    L:              np.1d array of length 3, DEFAULT np.ones( 3)
                    L dataset which will be added
    dset_names:     list of strings, default None
                    Name of the specified datasets to add _L dataset to
                    only use this if your datasets are not called "dset_i"
                    overwrits the previous "dset_nr" inputs and takes the specified
                    names in the given list instead
    Returns:
    --------
    None:           only adds the datasets to the hdf5 file
    """
    h5file = h5py.File( filename, 'r+')
    writing_directory = h5file[ internal_path]

    if dset_names is not None:
        for dset in dset_names:
            dset_name = dset + '_L'
            try:
                writing_directory.create_dataset( dset_name, data=L, compression='gzip')
            except:
                print( 'dataset {} already exists, continuing'.format( dset_name) )
    else:
        for i in range(start_nr, dset_nr+1):
            dset_name = 'dset_{}_L'.format( i)
            try:
                writing_directory.create_dataset( dset_name, data=L, compression='gzip')
            except:
                print( 'dataset {} already exists, continuing'.format( dset_name) )

