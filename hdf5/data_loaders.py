import numpy as np
import h5py
import sys
import os

class H5Loader():
  """
  Class with no functionality, simply has default methods for inheritance
  """
  def __init__( self, *args, **kwargs):
      pass

  def slice_to_ints( self, idx):
    if isinstance( idx, slice):
        if idx.start is None: start = 0
        elif idx.start < 0: start = self.n_samples + idx.start
        else: start = idx.start
        if idx.step is None: step = 1
        else: step = idx.step
        if idx.stop is None: 
            stop = self.n_samples if start >= 0 else 0
        elif idx.stop < 0: stop = self.n_samples + idx.stop
        else: stop = idx.stop
        idx        = list( range( start, stop, step) )
    elif isinstance( idx, int):
        idx = [idx]
    elif isinstance( idx, (list,tuple,np.ndarray)): 
        if isinstance( idx, np.ndarray) and idx.ndim != 1:
            raise Exception( 'only 1d-arrays are admissible for indexing {}'.format( self.__name__ ) )
    else: 
        raise Exception( 'Illegal indexing for {}, accepted are int/slice/list/tuple/np.1darray'.format( self.__name__))
    return idx

  def single_dset_loading( self, idx, getter):
      """
      Given the getter which contains all the information in the h5file
      load all datasets in 'idx' into a single array
      Loads each dataset individually since each dataset is stored individually
      Parameters:
      -----------
      idx:      list like of ints
                iterable of ints containing the indices
      """
      data = []
      for i in idx:
          data.append( getter(i) )
      data = np.concatenate( data, axis=0)
      return data


class Loader3d( H5Loader):
    """ quick and dirty fix to just get the 3d data in a structured manner """
    def __init__(self, path='/scratch/lissner/dataverse/rve_3d_tests/' ):
        self.path = path + '/' if  path[-1] != '/' else path 
        self.target_path = self.path + 'fixed_tangent_moduli.npz'
        self.filepath = self.path + '3d_rve.h5'
        self.h5path = '/image_data/dset_{}' 
        self.buggy_samples = [4,5,6,7] 
        self.n_samples = 2000 - len(self.buggy_samples)  #because 4 samples are buggy
        self.res = 3*(256, )
        self.replacement_idx = [2000 - x for x in self.buggy_samples]

    def open( self):
        self.file = h5py.File( self.filepath, 'r' )
        self.target_values = np.zeros( (self.n_samples, 21) )
        data_container = np.load( self.target_path)
        for i in range( self.n_samples):
            if i in self.buggy_samples:
                j = self.n_samples + i % self.buggy_samples[0]
            else:
                j = i 
            self.target_values[i] = data_container[f'arr_{j}'] #were stored in a npz file

    def close( self):
        self.file.close()
        del self.target_values

    def __getitem__( self, idx):
        """return images and targets """
        self.open()
        idx = self.slice_to_ints( idx)
        n_samples = len( idx)
        images = np.zeros( (n_samples, *self.res, 1), dtype='float32' )
        for i, j  in zip( range( n_samples), idx):
            if j in self.buggy_samples: 
                j = self.n_samples + j % self.buggy_samples[0]
            images[i,...,0] = self.file[self.h5path.format(j)][:]
        targets = self.target_values[idx]
        self.close()
        return images, targets
            



class DarusLoader(H5Loader):
  """ 
  Create an object which accesses the dataset uplaoded to darus
  The object can adjust which data is loaded exactly. It will always
  load the data into numpy arrays aranged row wise, such that each row
  is 1 sample. The returned data on indexing is always in this order:
  images, targets, features. Per default it loads images and target values 
  of 'both' inclusions types.
  Examples:
  ---------
  # load in default features and image/target data
  data_loader = DarusLoader( data_path='~/dataverse/', inclusion_type='circle')
  data_loader.load_features(True)
  images, kappa, features = data_loader[:n_samples]
  
  # only load in xi features and kappa (does not have to invoke a new object instance)
  data_loader.load_features( xi=20)
  data_lodaer.load_images( False)
  data_loader.set_inclusion( 'both' )
  kappa, features = data_loader[ 50: 50+n_samples] #NOTE: here we get 2*n_samples samples 
  #alternating circle/rectangle datasamples, i.e. 2 samples per dset_nr
  """
  def __init__( self, data_path='/scratch/lissner/dataverse/', inclusion_type='both', **load):
    """
    Initialize all path dependencies and specify default return values.
    Parameters:
    -----------
    data_path:      str, default '/scratch/lissner/dataverse/'
                    path to the hdf5 file, looks for '2d_microstructures.h5'
                    If a hdf5 file ending is given the file is taken instead
    inclusion_type: str, default 'both'
                    load data from which microstructures, accepts
                    'circle', 'rectangle' or 'mix/both'
                    Note that when choosing 'both' the indexing returns
                    twice as many samples as indexed (2*n_dataset)
    **load          kwargs with default arguments
                    what data to load
    load_images:    bool, default True
    load_targets:   bool, default True
    load_features   bool, default False
    """
    ##input preprocessing
    load_images = load.pop('load_images', True)
    load_targets = load.pop('load_targets', True)
    load_features = load.pop('load_features', False)
    h5_fname = '2d_microstructures.h5'
    if data_path[-3:] == '.h5' or data_path[-5:] == '.hdf5':
        h5_fname = ''
    if data_path[-1] != '/':
        data_path += '/'
    ## hardwired variable allocation (may change for different hdf5 files)
    self.__name__         = 'DarusLoader' #for console feedback below
    self.__len__          = 30000 #might want to move that to a method
    self.filename         = data_path + h5_fname
    self.h5file           = h5py.File( self.filename, 'r' )
    self.image_resolution = (400, 400)
    self.dset             = 'dset_{}'
    self.n_samples        = 15000
    ## preallocation of internal variables for loading
    self.load_images( load_images)
    self.load_targets( load_targets)
    self.load_features( load_features) 
    #self.load_pcf(False) #mb todo, depends on whether i want to store that
    self.set_inclusion( inclusion_type)


  #### setters for relevant parameters
  def set_inclusion( self, inclusion_type):
    """ choose between 'circle', 'rectangle' or 'mixed/both' inclusion types """
    inclusion_type = inclusion_type.lower()
    image_path   = '{}_inclusions/image_data/' #+dset_{}
    feature_path = '{}_inclusions/features/' #+xi or phi{0-9} +dset_{}
    target_path  = '{}_inclusions/effective_properties/heat_conductivity/' #+dset_{}
    if 'circ' in inclusion_type:
        inclusions  = ['circle']
        self.n_incl = 1
    elif 'rect' in inclusion_type:
        inclusions  = ['rectangle']
        self.n_incl = 1
    elif  'mix' in inclusion_type or 'both' in inclusion_type:
        inclusions  = ['circle', 'rectangle']
        self.n_incl = 2
        print( 'NOTE: {} will return twice the requested data when indexing. Giving the requested dataset numberes alternating for "circle" and "rectangle" inclusions'.format( self.__name__) )
    else:
        raise Exception( 'Non admissible parameter for inclusion_type passed. Make sure it contains the substring "circ", "rect" or "mix/both".' )
    self.image_path   = [image_path.format( x) for x in inclusions]
    self.feature_path = [feature_path.format( x) for x in inclusions]
    self.target_path  = [target_path.format( x) for x in inclusions]
 
  def set_inclusions( self, *args, **kwargs ):
      """ shadow set_inclusion to catch typos """
      return set_inclusion( *args, **kwargs)


  ##### Variable allocation for all loading switches
  def load_images( self, switch=False):
    """ switch to load store whether to return the images"""
    self.image_switch = switch
  
  def load_targets( self, switch=False, formatting='mandel'):
    """ formatting: choose between 'array/matrix', 'voigt', 'mandel' """
    self.target_switch = switch
    if formatting.lower() == 'mandel':
        self.target_format = lambda x: np.array( [x[0,0], x[1,1], 2**0.5 *x[0,1] ])
        self.target_shape  = (3,)
    elif formatting.lower() == 'voigt':
        self.target_format = lambda x: np.array( [x[0,0], x[1,1], 2*x[0,1] ])
        self.target_shape  = (3,)
    elif formatting.lower() == 'array' or formatting.lower() == 'matrix':
        self.target_format = lambda x: x
        self.target_shape  = (2,2)


  def load_features( self, switch=False, **which):
      """ 
      specify which features to load. If <switch> is True and which is 
      not specified then it gives all default values. If which is specified
      then only the specified <which> will be loaded
      On indexing all features will be concatenated into one array based 
      based on the pased order, default order given in the parameters
      Parameters:
      -----------
      switch:                   bool, default False
                                if True it takes all default kwargs which are not specified
      **which:                  kwargs with default arguments
        reduced_coefficients:   int, default 13
                                how many of the reduced coefficients to load
        xi:                     alias for reduced_coefficients
        #TODO NOT YET IMPLEMENTED manual: bool, default True
        dictionary_features:    int, default 8, values 0-8 admissible
                                which of the 9 phi sets to load 
        phi:                    alias for dictionary_features
      """
      if not which and switch is False:
        self.feature_switch = switch
        return
      self.feature_switch = True
      rb_warning = 'NOTE: xi coefficients will always be loaded from the "mix_rb" '
      phi_size = 2*[10] + 3*[20] + 4*[30] #size of each phi set
      if which:
          self.n_features = []
          self.features   = []
          for key, value in which.items():
              if 'xi' in key or 'coefficients' in key:
                  self.n_features.append( value)
                  self.features.append( 'xi/mix_rb/')
                  print( rb_warning)
              elif 'phi' in key or 'dictionary' in key:
                  ## NOTE very unclean workaround for the different lengthts
                  ## of the phis shadi provided
                  self.n_features.append( phi_size[value])
                  self.features.append( 'phi{}/'.format(value)) 
      else:
          self.n_features = [13 , phi_size[8]]
          self.features   = ['xi/mix_rb/', 'phi8/' ]
          print( rb_warning)


  ##### getter methods for single integer indexing
  def get_images( self, idx):
    """ get the requested dataset number, idx is an int"""
    images = np.zeros( (self.n_incl, *self.image_resolution ))
    for i in range( self.n_incl):
        images[i] = self.h5file[ self.image_path[i] + self.dset.format( idx) ]
    return images

  def get_targets( self, idx):
    """ get the requested dataset number, idx is an int"""
    targets = np.zeros( (self.n_incl, *self.target_shape ))
    for i in range( self.n_incl):
        targets[i] = self.target_format( self.h5file[ self.target_path[i] + self.dset.format( idx) ] )
    return targets

  def get_features( self, idx):
    """ get all the features based on the requested stuff above, idx is an int """
    features = np.zeros( (self.n_incl, sum(self.n_features) ) )
    for i in range( self.n_incl):
      s0 = 0 #index_start (too long to have fll variable name)
      for j in range( len(self.features) ):
          data = self.h5file[ self.feature_path[i] + self.features[j] +self.dset.format(idx)]
          features[i, s0:s0+self.n_features[j]] = np.squeeze( data)[:self.n_features[j]] 
          s0 += self.n_features[j]
    return features


  ##### dunder methods for indexing etc
  def __getitem__( self, idx):
    ## input formatting, format all types of inputs to list of ints
    idx = self.slice_to_ints( idx)
    data    = []
    getters = []
    ## allocate the defined order of returns and find out which are requested
    if self.image_switch is True:
        getters.append( self.get_images)
    if self.target_switch is True:
        getters.append( self.get_targets)
    if self.feature_switch is True:
        getters.append( self.get_features)
    ## for all indices get the requested image
    for getter in getters:
        data.append( self.single_dset_loading( idx, getter ) )
    if len( data) == 1:
        data = data[0]
    return data



class MixedLoader(H5Loader):
    """
    A Loader object to access the structure from the big hdf5 file, arranged from
    sanaths script which churns out the data. Works for any hdf5 file as long as 
    it follows the folder structure and naming convention (up to certain variations)
    If thermal and mechanical is loaded the number of samples is twice the number requested.
    They will be alternating, one mech, one thermal
    Regarding the data format, i.e. field, homogenized properties etc. are loaded for all specified
    'problems', can not differentiate between this
    Loads all data directly into memory, does not support to 'peek' into the file with a reference
    """
    def __init__( self, inclusions=['circle','rectangle'], data_file=None, problems=['mech'], **load):
        """
        Parameters:
        -----------
        inclusions: list of str, default ['rectangle', 'circle']
                    which inclusions to load
        problems:   list of str, default ['mech']
        data_file:  str, default None
                    full path to file, defaults to /scratch/lissner/...
        **load      kwargs of bools
                    specify which datatype to load. see 'set_getters' for reference
        """
        if data_file is None:
            self.filepath = '/scratch/lissner/dataverse/2d_mech+heat/2d_microstructures_results.h5'
        else:
            self.filepath = data_file
        ## properties wired into the file
        self.basepath = '{}_inclusions/dset_{}/Linear{}_hex8r/'
        self.image_path =  '{}_inclusions/dset_{}/image'
        self.feature_path =  '{}_inclusions/dset_{}/features'
        self.n_samples = 15000
        if '2d_new_microstructure' in self.filepath:
            self.n_samples = 1500
            inclusions = ['rectangle']
        ### make the input uniform to be used in the file
        self.set_problems( problems)
        self.set_inclusions( inclusions)
        ## allocations for the loading
        self.set_getters( **load)
        self.special_loading = False #only for heat of different phase contrasts


    def open( self):
        """ open the hdf5 file to read the data """
        self.file = h5py.File( self.filepath, 'r' )

    def close( self):
        """ close the file to free access"""
        self.file.close()

    def set_problems( self, problems):
        self.problems = []
        problems = [problems] if not isinstance( problems, (list, tuple)) else problems
        problems = ' '.join( problems).lower() 
        if 'therm' in problems or 'heat' in problems or 'temp' in problems:
            self.problems.append( 'Thermal')
        if 'mech' in problems or 'elast' in problems:
            self.problems.append( 'Elastic')

    def set_inclusions( self, inclusions):
        self.shapes =  []
        inclusions = [inclusions] if not isinstance( inclusions, (list, tuple)) else inclusions
        inclusions = ' '.join( inclusions).lower()
        known_convention = False
        if 'rect' in inclusions:
            self.shapes.append( 'rectangle')
            known_convention = True
        if 'circ' in inclusions:
            self.shapes.append( 'circle')
            known_convention = True
        if not known_convention:
            self.shapes.append( inclusions) 


    def set_getters( self,  **loading):
        """ 
        set all the getters which are called on indexing, or data return
        Here it is predefined which datasets will be loaded
        Defaults to: images=True, features=False, homo=True, field=False
        Data will be return in this order (most likely)
        """ 
        if len( loading) == 0:
            loading = dict( images=True, features=False, homo=True, field=False)
        self.getters = []
        for key in loading:
            if not loading[key]:
                continue
            key = key.lower()
            if 'field' in key:
                self.getters.append( self.get_fields)
            if 'hom' in key or 'target' in key:
                self.getters.append( self.get_homogenized)
            if 'image' in key or 'img' in key:
                self.getters.append( self.get_images)
            if 'feature' in key or 'fts' in key:
                self.getters.append( self.get_features)


    def set_phase_contrast( self, phase_contrast=[5], fname=None):
        """
        Set the phase contrast if more contrasts should be loaded besides only the
        fixed 'train' contrast. Does only work if heat data is loaded exclusively
        Parameters:
        -----------
        phase_contrast: int or list of ints
                        which phase contrasts to load on the getter
        fname:          str, default None
                        defaults to the data path of daework, the file has the name 'full_2d_data.h5'
        """
        assert not 'Elastic' in self.problems, 'can not load variable phase contrast for mechanical problem'
        phase_contrast = [phase_contrast] if not hasattr( phase_contrast, '__iter__') else phase_contrast
        if phase_contrast == [5]:
            self.special_loading = False
            return
        self.special_loading = True
        self.kappa_file = '/scratch/lissner/dataverse/2d_rve_kappa_contrast/full_2d_data.h5' if  fname is None else fname
        self.kappa_path = '{}/target_values/heat_conduction/contrast_{}/dset_{}'
        self.phase_contrast = phase_contrast

    def get_fields( self, idx):
        feature_path = 'stress_load{}'
        independent_components = ( [0,1,2], [1,2], [2] )
        data = [[], []]
        for i in idx:
          for j, problem in zip( range( len(self.problems)), self.problems):
            assert problem != 'Thermal', 'sorry, thermal field loading not yet implemented :P' 
            for incl in self.shapes:  #alternating inclusions
              fields = []
              for k, loadings in enumerate( independent_components):
                  load_case = self.file[ self.basepath.format( incl, i, problem) + feature_path.format(k)]
                  for l in loadings:
                    fields.append( load_case[l] )
              data[j].append( self.reformat_fields( np.stack( fields, axis=0) ) )
        if len( self.problems) == 1:
            return np.stack( data[0], axis=0 )
        return [np.stack( x, axis=0) for x in data ]


    def get_features( self, idx):
        data = []
        for i in idx:
          for incl in self.shapes:  #alternating inclusions
            features = self.file[ self.feature_path.format( incl, i)][:]
            data.append( features )
        data = np.stack( data, axis=0)
        if self.special_loading: #replicate the features and add the phase contrast everywhere
            multi_contrast = [] 
            for contrast in self.phase_contrast:
                multi_contrast.append( np.insert( data, 1, contrast, axis=1))
            data = np.concatenate( multi_contrast, axis=0)
        return data


    def get_homogenized( self, idx):
        """
        load in all homogenized response given the previously defined
        parameters
        """
        if not self.special_loading:
          feature_path = 'hom_response'
          data = [[], []]
          for i in idx:
            for j, problem in zip( range( len(self.problems)), self.problems):
              for incl in self.shapes:  #alternating inclusions
                feature = self.file[ self.basepath.format( incl, i, problem) + feature_path][:]
                data[j].append( self.reformat_homo( feature, problem) )
          if len( self.problems) == 1:
              return np.stack( data[0], axis=0 )
          return [np.stack( x, axis=0) for x in data ]
        else: #variable phase contrast loading
          h5file = h5py.File( self.kappa_file, 'r' ) 
          data = [ [] for _ in self.phase_contrast]
          for i in idx:
            dset = i // 1500 + 1
            i = i % 1500
            for j, contrast in zip( range( len(self.phase_contrast) ), self.phase_contrast)  :
              for incl in self.shapes:  #alternating inclusions
                  feature = h5file[ self.kappa_path.format( incl, contrast, dset )][:,i]
                  data[j].append( feature)
          h5file.close()
          data = np.concatenate( [np.stack( x, axis=0) for x in data], axis=0)
          data[:,-1] *= -1
          return data

    def get_images( self, idx):
        data = []
        for i in idx:
          for incl in self.shapes:  #alternating inclusions
            image = self.file[ self.image_path.format( incl, i)][:]
            data.append( self.reformat_fields( image) )
        return np.stack( data, axis=0)

    
    def reformat_homo( self, homo_matrix, problem):
        """
        Reformat the homogenized response given in matrix format to 
        mandel notation
        """
        if problem == 'Elastic':
            kappa = np.zeros( 6)
            ## full 3x3 representation (does contain symmetries)
            #for i in range(3): #3 responses
            #  for j in range( 3): #3 loadings
            #    prefactor = 2**0.5 if i == 2 else 1
            #    kappa[i+j*3] = prefactor * homo_matrix[i,j]
            ## representation using symmetries and mandel notation
            diag_weights = [1,1,2]
            for i in range(3):
                kappa[i] = diag_weights[i]*homo_matrix[i,i]
            kappa[-3] = homo_matrix[0,1]
            kappa[-2] = 2**0.5* homo_matrix[0,2]
            kappa[-1] = 2**0.5* homo_matrix[1,2]
        if problem == 'Thermal':
            kappa = np.zeros( 3)
            for i in range( 2):
                kappa[i] = homo_matrix[i,i]
            kappa[-1] = 2**0.5* homo_matrix[0,1]
        return kappa

    def mandel_to_tensor( self, data, problem ):
        """ 
        reformat the given data which has been vectorized in mandel 
        notation and put it back into tensor notation with the correct 
        weighting. Blindly assumes the 2d case for now
        Parameters:
        -----------
        data:   numpy 2d-array
                vectorized data array in mandel notation
        """
        if problem == 'heat':
            tensor = np.zeros(( data.shape[0], 2,2))
            tensor[:,0,1] = data[:,-1] / 2**0.5
            tensor += np.transpose( tensor, axes=(0,2,1) )
            for i in range( 2):
                tensor[:,i,i] = data[:,i]
        elif problem == 'mech':
            tensor = np.zeros(( data.shape[0], 3, 3))
            for i,j in enumerate( [(slice(None), 1,2), (slice(None), 0,2), (slice(None), 0,1)] ):
                tensor[j] = data[:,i-1] / 2**0.5
            tensor += np.transpose( tensor, axes=(0,2,1) )
            for i in range( 3):
                tensor[:,i,i] = data[:,i]
        return tensor
    
    def reformat_fields( self, image):
        """
        Given any flattened image, reformat it to original resolution and
        tensorflow format, i.e. n_samples x res x res x n_channels
        """
        res = int( round( image.shape[-1]**0.5) )
        n_channels = image.shape[0]
        images = np.zeros( (res, res, n_channels) )
        for i in range( n_channels):
            images[...,i] = image[i].reshape( res, res)
        return images


    def __getitem__( self, idx):
      """ return all the requested data upon indexing """
      ## input formatting, format all types of inputs to list of ints
      idx = self.slice_to_ints( idx)
      data    = []
      ## allocate the defined order of returns and find out which are requested
      try:
          self.open()
      except:
          if not os.path.exists( self.filepath):
            print( f'file {self.filepath} does not exist, unable to load data' )
          else:
            print( 'file already open for reading, closing it after loading' )
      for getter in self.getters:
          data.append( getter( idx) )
      ## for all indices get the requested image
      if len( data) == 1:
          data = data[0]
      self.close()
      return data 
