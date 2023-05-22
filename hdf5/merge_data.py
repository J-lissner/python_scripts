import numpy as np
import h5py


def list_to_hdf5( input_data, filename, h5_path, write_name='dset_{}', writing_number=None, metadata=None):
    """
    Takes data written into a list (input_data) and writes it into the "filename"
    on the specified internal path. It is assumed that each entry in the list is
    one numbered dataset in consecutive order. Each entry in the list will be 
    one dataset, datatype is implicitely taken by h5py.
    Parameters:
    -----------
    input_data:     list (of e.g. numpy arrays)
                    multiple datasets stored in the list-like
    filename:       string
                    location and filename of the hdf5 file
    h5path:         string
                    Internal path to the hdf5 file where the data will be stored

    write_name:         formatted string, default 'dset_{}'
                        name of every dataset, they will only deviate in the 
                        number (formatted in string)
    writing_number:     int, default None
                        specifies the indices of the formatted string, if not 
                        given the datasets are numbered from 0 to len( input_data)-1
                        otherwise it goes from "writing_number" to 
                        "writing_number + len( input_data)-1 "
    metadata:           dictionary, default None
                        attached metadata to the "h5path", only writes if given
    Returns:
    --------
    None:       only writes data into the target file
    """
    h5file = h5py.File( filename, 'a')
    try:
        writing_directory = h5file.create_group( h5_path)
    except:
        writing_directory = h5file[ h5_path]

    if not metadata is None:
        writing_directory.attrs.update( metadata)

    if not writing_number is None:
        counter = writing_number
    else: 
        counter = 0 

    for i in range( len( input_data) ): 
        writing_directory.create_dataset( write_name.format( i+counter), data=input_data[i], compression='gzip', compression_opts=9 )
    h5file.close()


def hdf5_to_hdf5( read_file, read_location, read_numbers, write_file, write_location=None, write_number=None, read_name='dset_{}', write_name=None, metadata=True):
    """
    Takes datasets out of "read_file[read_location]" and merges them into 
    the "write_file[write_location]".  Assumes that the datasets in the 
    "read_location" share the same name but are numbered
    Returns nothing, only writes into the specified hdf5 file
    Parameters:
    -----------
    read_file:      string  
                    path and filename to the reading file
    read_location:  string  
                    internal path in the hdf5 file
    read_numbers:   int or list of two int  
                    Careful, if "write_number" is not given they are copied
                    numbering of the datasets, if 1 int is given it numbers 
                    them from 0 to read_numbers-1 if a list of int is given it 
                    reads the numbers from NR[0] to NR[1] and NR[1] included
    write_file:     string  
                    path and filename to the file to write in
    write_location: string, default None  
                    internal path in the hdf5 file, copies the old path if
                    no value is given.
    write_number:   int, default None
                    If None are given, the "read_numbers" are copied
                    If an int is given, it writes the numbers from "write_number" upward
    read_name:      formatted string, default 'dset_{}'
                    shared name of the datasets with the numbering indices for reading,
                    String must share the same format as the default
    write_name:     formatted string, default None
                    shared name of the datasets with the numbering indices for the writing,
                    if no argument is given, it copies the name of the "read_dataset"
    metadata:       Boolean or dict, default True
                    If some metadata should be attached to the "write_location"
                    if metadata=True: it tries to attach any metadata from "read_locaion"
                    if metadata=False: no metadata is attached
                    If metadata is a dictionary, the specified metadat ais attached to "write_location"
    Returns:
    --------
    None:       only writes data into the target file
    """
    ### Default parameters
    if write_name is None:
        write_name = read_name
    if write_location is None:
        write_location = read_location
    if write_number is None:
        if isinstance( read_numbers, list):
            write_number = read_numbers[0]
        elif isinstance( read_numbers, int):
            write_number = 0

    ### hdf5 files and file creation
    storage_file = h5py.File( write_file, 'a')
    data_file =  h5py.File( read_file, 'r' )
    read_dir = data_file[ read_location]
    try:
        writing_dir = storage_file[ write_location]
    except:
        writing_dir = storage_file.create_group( write_location)
    if metadata is True:
        writing_dir.attrs.update( read_dir.attrs)
    elif not metadata is False:
        writing_dir.attrs.update( metadata)

    ### Writing of data
    if isinstance( read_numbers, int):
        for i in range( read_numbers):
            counter = write_number +i 
            data = read_dir[ read_name.format(i) ]
            writing_dir.create_dataset( write_name.format( counter), data=data, compression='gzip', compression_opts=9)
    elif isinstance( read_numbers, list ):
        for i in range( read_numbers[0], read_numbers[-1]+1 ):
            print( 'writing for number', i)
            counter = i - read_numbers[0]+ write_number
            data = read_dir[ read_name.format( i) ]
            writing_dir.create_dataset( write_name.format( counter), data=data, compression='gzip', compression_opts=0)
    else: 
        raise Exception( 'Non conform input arguments for the indices (read/write_numbers), unable to write -> terminating program, Note that this does not work for numpy arrays for whatever reason')
    data_file.close()
    storage_file.close()


    
