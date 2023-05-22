import h5py
import os

def chunked_loading(data_container, idx=None ):
    """
    Knowing the chunks of the image dataset it is faster to manually load
    each chunk separately and concatenate it afterward. This method does
    exactly that. Basically returns all requested datasets in idx. CARE:
    It assumes that the datasets are given in tensorflow notation with 
    samples in the first index, and that is where chunks are laid out, 
    always taking full samples. No performance guarantees can be made 
    for differently chunked data.
    Parameters:
    -----------
    data_container: path do h5py.dataset
                    index acessible reference to the dataset
    idx:            iterable of ints with 'index method' (e.g. lists)
                    requested indices of the dataset
    """
    if idx is None:
        idx = list( range( data_container.shape[0] ) ) 
    ## input preprocessing
    chunksize = data_container.chunks[0]
    sorting = np.argsort( idx ).squeeze()
    idx = idx[sorting]
    n_chunks = ceil((idx[-1]-idx[0])/chunksize )
    data = []
    i = 0
    j = 0
    while i < len( idx ):
        chunk_index = [idx[i]]
        current_chunk = idx[i] // chunksize
        i += 1 #adding first index
        while i < len(idx) and (idx[i] // chunksize) == current_chunk:
            chunk_index.append( idx[i] ) #indices of current chunk
            i += 1 #added another index
        data.append( data_container[chunk_index] ) #load the single chunk
        j+= 1
    data = np.concatenate( data, axis=0 )
    return data[np.argsort(sorting)] #undo sorting to requested order



def display_file_contents(filename):
    h5file = h5py.File( filename, 'r')
    display_all_data(h5file)
    h5file.close()

def display_all_data(h5file):
    """ 
    Combines all display-functions to display every stored dataset and metadata
    Input: h5file - opened file object of h5py
    Output: Terminal Output, every data in the file
    """
    print('\n#### Datasets and metadata in root folder ####' )
    display_root(h5file) #displays only datasets and 
    print('\n#### Datasets stored in each folder ####' )
    h5file.visititems(datasets)
    # display only the metadata/attributes
    print('\n#### Metadata stored in each folder ####' )
    h5file.visititems(attrs)
    
def attrs(name, file_obj):
    """
    Function specifically created for the h5py.File.visititems(func) command
    USAGE: call with h5py.File.visititems(print_attrs)
    Output: Terminal output, every metadata and which file it is attached to
    """
    if file_obj.attrs.keys():
        print( '\nMetadata attached to:', name)
        for key in file_obj.attrs.keys():
            print("    %s: %s" % (key, file_obj.attrs[key]) )


def datasets(name, file_obj):
    """
    Function specifically created for the h5py.File.visititems(func) command
    USAGE: call with h5py.File.visititems(print_datasets)
    Output: Terminal output, every dataset and in which folder it is located
    """
    try:
        dirs = list(file_obj.keys())
        i = 0
        found_ds = False
        while i < len(dirs) and not found_ds:
            found_ds = isinstance(file_obj[dirs[i]], h5py.Dataset)
            i += 1

        if found_ds:
            print( '\ncontent of folder: %s' % name)
            [print(file_obj[datasets]) for datasets in dirs if isinstance(file_obj[datasets], h5py.Dataset)]
    except:
        pass


def display_root(h5file):
    """
    Function for hdf5 files, used to display all datafiles and metadata in the root folder
    Does ignore all folders on output!

    Combines all display-functions to display every stored dataset and metadata
    Input: h5file - opened file object of h5py
    Output: Terminal Output, every data in the root folder
    """
    try:
        dirs = list( h5file.keys() )
        i = 0
        found_ds = False
        while i < len(dirs) and not found_ds:
            found_ds = isinstance(h5file[dirs[i]], h5py.Dataset)
            i += 1

        if found_ds:
            print('\nDatasets of the root directory:')
            [ print(h5file[x]) for x in dirs if isinstance(h5file[x], h5py.Dataset)]
        if h5file.attrs.keys():
            print('\nAttributes/Metadata in the root directory:')
            for metakeys in h5file.attrs.keys():
                print('    {}: {}'.format( metakeys, h5file.attrs[metakeys]) )
    except: 
        pass


def file_size(fname):
    """
    Shows the size of the file "fname"
    Input:  fname - absolute or relative location of the file to inspect
    Output: returns string - size of the file in '*iB' format
    """
    if os.path.isfile(fname):
        file_info = os.stat(fname)
        return convert_bytes(file_info.st_size)

def convert_bytes(num):
    """
    Convert bytes into the largest possible format with num > 1
    Input: num - filesize in bytes
    Output: returns string - formatted filesize in *'iB' format 
    The *iB refers to the binary file size format
    This function is mainly used by the file_size(fname) function
    """
    for x in ['bytes', 'KiB', 'MiB', 'GiB', 'TiB']:
        if num < 1024.0:
            return  "%3.2f %s" % (num, x) 
        num /= 1024.0


#### newly added stuff (gotten from sanath)
def allkeys(obj, keys=[]):
  ''' Recursively find all keys '''
  keys.append(obj.name)
  if isinstance(obj, h5py.Group):
    for item in obj:
      if isinstance(obj[item], h5py.Group):
        allkeys(obj[item], keys)
      else: # isinstance(obj[item], h5py.Dataset):
        keys.append(obj[item].name)
  return keys
#for key in key_list:
#    metadata = dict( h5[key].attrs)
#    if metadata:
#        print()
#        print( 'folder/dset:', key)
#        for descriptor, parameter in metadata.items():
#            print( descriptor, ':', parameter)
#    #if isinstance(h5[key], h5py.Dataset):
#    #    globals()[key.replace('/','_')] = np.array(h5[key])





### newly added stuff from me
def show_folderstructure( h5file):
    """
    Show the folder structure of the deepest branch of the h5file
    also shows folders containing datasets with a leading '*'
    NOTE this function is buggy somewhere, though the file is too big to debug
    Parameters:
    -----------
    h5file:     string or h5py.File
                identifier to hdf5 file
    Returns:
    --------
    paths:      list of strings
                all paths found in the hdf5 file
    paths:      list of strings
                all paths which contain datasets
    """
    if isinstance( h5file, str):
        h5file = h5py.File( h5file, 'r' )
        close = True
    else:
        close = False
    items = allkeys( h5file)
    paths = set()
    for item in items:
        paths.add( os.path.dirname( item) )
    paths = list( paths) 
    j = 0
    ## check if i have a leaf
    while j < len(paths): 
        branch = paths.pop( j)
        leaf = True
        for path in paths:
            if branch in path: #if the branch is a substring of path (not a final branch)
                leaf = False
                break
        if leaf is True:
            paths.insert( j, branch)
            j += 1 
        
    datapaths = []
    for h5path in paths:
        for item in h5file[ h5path]:
            try: #might be required if datasets are in root
                if isinstance( h5file[h5path +'/'+ item], h5py.Dataset ):
                    datapaths.append( '*' + h5path  )
                    break 
            except:
                if isinstance( h5file[h5path + item], h5py.Dataset ):
                    datapaths.append( '*' + h5path  )
                    break 
    if close is True:
        h5file.close()
    return paths, datapaths


