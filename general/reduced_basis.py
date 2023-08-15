import numpy as np
from numpy.fft import fftn, ifftn
import time 

class ReducedBasis():
    """
    A reduced basis object which can iteratively update the reduced basis
    given snapshot data. Can deploy three different methods
    Has some additional functionality for the 2-point correlation function
    derived from image data and tracks variables thereof which can be ignored
    """
    def __init__( self, computation_method='svd', track=True, resolution=(400,400), scaling='fscale'):
        """ 
        Initialize the Class with some default parameters,
        Initialized with self.set_limits() and self.set_S_sizes() with default parameters
        Parameters:
        -----------
        computation_method: str
                            choose between 'svd','corr_matrix' and expand
                            the last one yields a bigger basis faster
        resolution:         list of ints, default (400,400)
                            resolution of the vectorized snapshots
        track:              bool, default True
                            whether metrics to be tracked should be initialized 
        scaling:            str, default 'fscale'
                            which scaling to implement on the 2pcf, see
                            self.scale_snapshots for reference
        """
        allowed_methods = [ 'svd', 'corr_matrix', 'expand' ]
        if computation_method in allowed_methods:
            self.computation_method = computation_method.lower()
        else:
            raise Exception("Non allowed method, choose between %s" % allowed_method )
        self.scaling = scaling
        self.resolution = resolution
        self.set_limits()
        self.set_S_sizes()
        if track:
            self.initialize_tracking_variables()


    #################################################################
    ### bulk of the computation, basis computation and adjustment ###
    def compute_basis( self, S, store_snapshots=True ):
        """ 
        Input: S: Snapshot matrix, column wise arranged
        store_snapshots:    bool, default True
                            wether or not to store the snapshots used for 
                            original computation, just implemented for 
                            convenience right now in the post processing
        Returns: self.B: reduced basis
                 self.Theta: sorted eigenvalues 
                (self.V: only for 'corr_matrix', eigenvectors of the snapshot correlation matrix)
        """
        self.s_orig = S if store_snapshots else None
        if self.computation_method == 'svd':
            U, Theta, W = np.linalg.svd( S, full_matrices=False )
        elif self.computation_method in ['corr_matrix', 'expand']:
            C_s = S.T @ S 
            C_s = 0.5* (C_s.T + C_s)
            Theta, V = np.linalg.eig(C_s)
            Theta =  (np.maximum( 0, Theta)**(0.5)).real 
            permutation = Theta.argsort()[::-1]
            Theta = Theta[permutation]
            V = V[:,permutation].real

        # Compute the truncation limit
        sum_eig = sum(Theta**2)
        for N in range( 1, S.shape[1]):
            trunc_error = max( 0, 1- np.sum(Theta[:N]**2)/sum_eig )**0.5
            if trunc_error < self.truncation_limit:
                break 
        # compute and truncate the RB
        self.Theta = Theta[:N]
        if self.computation_method == 'svd':
            self.B = U[:,:N]
        elif self.computation_method in ['corr_matrix', 'expand']:
            self.V = V[:,:N]
            self.B = S @ (self.V *1/self.Theta)


    def adjust_basis( self, delta_S): 
        """
        adjust the current basis with the snapshots matrix delta_S
        chooses between the method set in the __init__() call
        truncates the additional modes according to the truncation limit set with set_limits
        delta_S - Snapshot matrix for enrichment
        """
        n_s = delta_S.shape[1]
        N_rb = self.B.shape[1]
        if self.computation_method == 'expand':
            return self.expand_basis( delta_S)
        elif self.computation_method == 'svd':
            xi = self.B.T @ delta_S
            U_residual, Theta_residual, W_residual = np.linalg.svd( delta_S - self.B @ xi, full_matrices=False ) #svd( S_residual) after projection onto rb
            U, Theta, _ = np.linalg.svd(  np.vstack( (
                                 np.hstack( (np.diag(self.Theta),    xi) ), 
                                 np.hstack( (np.zeros( (n_s, N_rb)), np.diag(Theta_residual) @ W_residual)  )) ), full_matrices=False )

        elif self.computation_method == 'corr_matrix':
            S_approx = self.B * self.Theta @ self.V.T
            V_expanded = np.vstack( ( #this one is needed for computation of self.V
                                 np.hstack( (self.V,                 np.zeros( (self.V.shape[0], n_s)) )),
                                 np.hstack( (np.zeros( (n_s, N_rb)), np.eye(n_s)) )) )
            C_new = delta_S.T @ delta_S 
            C_new = 0.5* (C_new.T + C_new)
            C_oldnew = self.Theta[:,np.newaxis] * (self.B.T @ delta_S)
            C_star = np.vstack( (
                                 np.hstack( (np.diag(self.Theta**2), C_oldnew) ), 
                                 np.hstack( (C_oldnew.T,             C_new) )) )
            C_star = 0.5* (C_star.T + C_star)
            Theta, V_star = np.linalg.eig(C_star)
            Theta =  np.maximum( 0, Theta)**(0.5).real 
            permutation = Theta.argsort()[::-1]
            Theta = Theta[permutation]
            V_star = V_star[:,permutation]

        #compute the truncation limit
        added_info = np.linalg.norm( delta_S, 'fro')**2
        for N_trunc in range( delta_S.shape[1], -1, -1): #revert the range and loop until 1
            truncation_error = max( 0, np.sum( Theta[-N_trunc:]**2) /added_info )**0.5
            if truncation_error < self.truncation_limit:
                break 
        ## reassemble the basis update
        N_trunc = -N_trunc if N_trunc != 0 else None #for indexing below
        self.Theta = Theta[:N_trunc]
        if self.computation_method == 'svd':
            self.B = np.hstack( (self.B, U_residual) ) @ U[:,:N_trunc]
        elif self.computation_method == 'corr_matrix':
            self.V = V_expanded @ V_star[:,:N_trunc]
            self.B = np.hstack( (S_approx, delta_S) ) @ (self.V* (1/self.Theta) ) 


    def expand_basis(self, delta_S ):
        """
        Expands the basis by simply appending modes and leaving the existing ones unchanged
        Input:
        delta_S - Snapshots which will enrich the basis B
        Output:
        Enriched basis
        """
        S_residual = delta_S - self.B @ (self.B.T @ delta_S)
        C_s = S_residual.T @ S_residual
        C_s = 0.5* (C_s.T + C_s)
        Theta, V = np.linalg.eig(C_s)
        Theta = np.maximum( 0, Theta)**(0.5).real
        permutation = Theta.argsort()[::-1] 
        Theta = Theta[permutation]
        V = V[permutation] 
        # Compute the truncation limit
        sum_eig = np.linalg.norm( delta_S, 'fro')**2
        for N in range( delta_S.shape[1], -1, -1): #revert the range and loop until 1
            truncation_error = max( 0, np.sum( Theta[-N:]**2)/sum_eig )**0.5
            if truncation_error < self.truncation_limit:
                break 
        # truncate delta_B and return the expanded basis
        Theta = Theta[:N]
        V = V[:,:N]
        delta_B = S_residual @ (V *1/Theta)
        self.B = np.hstack( (self.B, delta_B) )
        self.Theta = np.hstack( (self.Theta, Theta) )
        #if self.computation_method == 'corr_matrix':
        #    self.V = np.hstack( (self.V, V) )

    #####################################
    ### functionalities with return value 
    def correlation_function( self, images, scale=True):
        """
        Compute the 2pcf of the flattened image arrays and (optional) scale
        them with the same scheme used for the reduced basis
        Assumes row wise arranged flattened data which is reshaped to
        the original resolution (stored in self)
        Parameters:
        -----------
        images: numpy nd-array of shape np.prod( self.resolution), n_s
                flattened and row wise arranged image data
        scale:  bool, default True
                whether the basis related scaling should be applied 
        Returns:
        --------
        pcf:    numpy nd_array
                column wise arranged 2pcf in Fourier space
        """
        n_s = images.shape[-1]
        pcf = np.zeros( images.shape, dtype=float )
        for i in range(n_s):
            sample_pcf = fftn( images[:,i].reshape( self.resolution) )
            sample_pcf = np.conj(sample_pcf) * sample_pcf  #real up to roundoff errors
            sample_pcf = ifftn(sample_pcf)
            pcf[:,i] = sample_pcf.real.flatten()
        if scale:
            pcf = self.scale_snapshots( pcf)
        return pcf

    def scale_snapshots(self, pcf):
        """ 
        Scale the snapshots of the 2pcf in real space given different 
        saling schemes. Applies a minimum 0 mean scaling. Choose between
        None: (or baseline) zero mean each snapshot
        'max1':   scale every corner to have
        'fscale': filter out the volume fraction by division
        Parameters:
        -----------
        pcf:    numpy 2d-array of shape (?, n_s)
                sample wise arranged data of snapshots
        Returns:
        --------
        pcf:    numpy 2d-array of shape (?, n_s)
                sample wise shifted (and scaled) snapshots
        """
        vol = np.mean( pcf, 0 )**0.5
        pcf = pcf - vol**2 #zero mean
        if self.scaling == 'max1': #every corner has value 1
            pcf = pcf / ( vol -vol**2 )
        elif self.scaling == 'fscale': #take out the volume fraction
            pcf = pcf / vol  
        return pcf


    def get_xi( self, pcf, n_xi=None):
        """
        compute the reduced coefficients of the snapshots
        using the current reduced basis
        Parameters:
        -----------
        pcf:    numpy 2d-array of shape (?, n_s)
                snapshot data (preferably used for basis computation
        n_xi:   int, default None
                how many eigenmodes to consider for projection
        Returns:
        --------
        xi:     numpy 2d-array of shape (n_xi, n_s)
                reduced coefficients of the snapshots
        """
        return self.B.T[:n_xi] @ pcf

    def mean_projection_error( self, pcf, n_xi=None):
        data_norm = np.linalg.norm( pcf,'fro')**2
        xi = self.get_xi( pcf, n_xi=n_xi )
        p_delta = (1.0- (np.linalg.norm( xi, 'fro')**2 / data_norm) )**0.5
        return p_delta


    def reorthogonalize_B( self, tolerance=10**-9):
        """
        Reorthogonalizes the reduced basis self.B 
        if B.T @ B = np.eye() is vialoted by a certain treshold
        """
        ortho = self.B.T @ self.B
        deviation = sum( abs( np.diag(ortho) -1) )
        if deviation > tolerance:
            Lower = np.linalg.cholesky( ortho)
            self.B = self.B @ np.linalg.inv( Lower).T
            return 1
        return 0


    ########################################
    ### allocators and tracker functions ###
    def set_limits( self, truncation_limit=0.05, allowed_pdelta=0.05, n_converged=150):#hierfür vllt noch getter schreiben
        """
        Sets the "accuracy limits" for the truncation and enrichment
        Input
        truncation_limit - percentage of information to discard on basis computation
        allowed_pdelta   - projection error for which snapshots to buffer for enrichment
        n_converged      - number of consecutive snapshots with pdelta < allowed_pdelta for convergence criteria
        """
        self.truncation_limit = truncation_limit
        self.allowed_pdelta = allowed_pdelta
        self.n_converged = n_converged

    def set_S_sizes( self, n_orig=50, n_adjust=25 ): #hierfür vllt noch getter schreiben
        """
        sets the sizes of the snapshots matrices for the basis computations/enrichments
        n_orig   - number of snapshots for original basis construction
        n_adjust - number of snapshots for adjustment of the basis
        """
        self.n_orig = n_orig
        self.n_adjust = n_adjust

    def initialize_tracking_variables( self):
        """
        Initialize some tracking variables, these include:
        n_updates       - number of updates conducted
        n_reortho       - list, when the reorthogonalization was done
        N_em            - list, how many eigenmodes after the current enrichment step
        n_pdelta_S_orig - list, current projection error on S_0
        n_over          - list, accumulated number of snapshots over the tolerance for the enrichment
        n_under         - list, discarded number of snapshots unter the tolerance for the enrichment
        n_reortho       - list, if a reorthogonalization has been conducted for this enrichment step
        """
        self.n_updates = 0
        self.iteration_time = [time.time()]
        self.n_reortho = []
        self.N_em = []
        self.pdelta_S_orig = []
        self.n_over = []
        self.n_under = []
        self.n_reortho = []
        self.tracked_variables = dict() #other variables which are tracked from remote

    def update_tracking_variables( self, n_over=None, n_under=None, pdelta=None, reorthogonalize=True, printout=False, **tracked ):
        """ 
        update the tracking variables defined by "initialize_tracking_variables()
        every manually given tracking variable is optional
        pdelta:     float, default None
                    if not given, it attempts to compute it on the original snapshots
                    as percentage value
        printout:   bool, default True
                    whether or not the data should be printed at call
        **tracked:  kwargs,
                    which other variables to track while training. Will store them in 
                    self.tracked_variables[ key] = [value1, value2,...]
        additional input: 
        reorthogonalize - if a reorthogonalization is to be carried out with the "update_tracking_variables" call
        """
        for key, value in tracked.items():
            if key not in self.tracked_variables:
                self.tracked_variables[key] = [value]
            else:
                self.tracked_variables[key].append( value)
        self.n_updates += 1
        self.iteration_time.append( time.time() - self.iteration_time[-1] )
        self.N_em.append( self.B.shape[1] )
        if not isinstance( pdelta, float) and self.s_orig is not None:
            pdelta = 100*self.mean_projection_error( self.s_orig )
        self.pdelta_S_orig.append( pdelta )
        self.n_over.append( n_over )
        self.n_under.append( n_under )
        if reorthogonalize == True:
            self.n_reortho.append( self.reorthogonalize_B() )
        if printout:
            print( f'    Current size of basis {self.N_em[-1]} after {self.n_updates} updates, projection error {100*pdelta:2.3f}' )
            print( f'    discarded {self.n_under[-1]} snapshots in the last iteration, {sum(self.n_under)} total' )
            print( f'    maximum number of discareded snapshots in a single update increment: {max( self.n_under) }' )




class FourierBasis(ReducedBasis):
    """
    Fourier basis is a special case of a Reduced Basis, computed in 
    "fourier space" (transformed to with fft)
    Has an additional function to truncate the eigenmodes to for 
    compuptational speedup and storage efficiency
    Tracks one more variable for debugging purposes
    """
    def __init__( self, computation_method='svd', *args, **kwargs): 
        #explicitely writing an init method seems better (using the super method)
        super().__init__( computation_method, *args, **kwargs)
        self.corner_indexing = slice( None)

    def corner_truncation( self, tol=0.01, inspected_mode=-1, shrink_basis=False):
        """ 
        Truncate all eigenmodes of the Fourier basis to the same size 
        governed by the error introduced in 'inspected_mode'
        After finding the thresold, the truncation is applied to the basis,
        such that it gets reduced. The basis is reweighted such that the 
        product of the reduced coefficients does not change except for the
        <tol> error introduced.
        This is doable since the fft is symmetric
        The function
        Parameters:
        -----------
        tol:            float, default 0.01
                        Maximum amount of "information" discarded
        inspected_mode: int, default -1
                        assuming increasingly higher frequencies, higher modes
                        require a larger corner
        shrink_basis:   bool, default False
                        whether or not to shrink the actual basis
        """
        if self.B.shape[1] < inspected_mode:
            inspected_mode = -1
        reference_mode = self.B[:,inspected_mode].reshape( self.resolution)
        ndim = len( self.resolution )
        info_last = np.linalg.norm( reference_mode, 'fro' )
        for N_c in range( 1, min(self.resolution)//2 -1):
            truncated_mode = reference_mode.copy().reshape( self.resolution)
            for i in range( ndim ): #ndim basically
                indexing = ndim*[slice(None)]
                indexing[i] = slice( N_c +1, self.resolution[i]-N_c)
                truncated_mode[ tuple( indexing) ] = 0
            diff = np.linalg.norm( reference_mode, 'fro') - np.linalg.norm( truncated_mode, 'fro') 
            #diff = np.linalg.norm( reference_mode - truncated_mode, 'fro')
            rel_error = diff /info_last
            if rel_error < tol:
                break
        self.N_c = N_c
        if not shrink_basis:
            truncated_mode = truncated_mode.flatten()
            self.corner_indexing = truncated_mode != 0
            self.B[self.corner_indexing] = 0
        else: #shrink the basis to the reduced size
            truncated_mode[ N_c+1:] = 0 #the rest is hermitian symmetric
            truncated_mode = truncated_mode.flatten()
            self.corner_indexing = truncated_mode != 0
            self.B = 2*self.B[ self.corner_indexing] #multiply by 2 to account for symmetries
            #rescale the 0th freqeuncy on all axis since this is not symmetrically copied
            self.B[0] = self.B[0]/2 


    #####################################
    ### functionalities with return value 
    def correlation_function( self, images, scale=True):
        """
        Compute the 2pcf of the flattened image arrays and (optional) scale
        them with the same scheme used for the reduced basis
        Assumes row wise arranged flattened data which is reshaped to
        the original resolution (stored in self)
        Parameters:
        -----------
        images: numpy nd-array of shape np.prod( self.resolution), n_s
                flattened and row wise arranged image data
        scale:  bool, default True
                whether the basis related scaling should be applied 
        Returns:
        --------
        pcf_four:   numpy nd_array
                    column wise arranged 2pcf in Fourier space
        """
        n_s = images.shape[-1]
        pcf_four = np.zeros( images.shape, dtype=float )
        for i in range(n_s):
            sample_pcf_four = fftn( images[:,i].reshape( self.resolution) )
            sample_pcf_four = np.conj(sample_pcf_four) * sample_pcf_four  #real up to roundoff errors
            pcf_four[:,i] = sample_pcf_four.real.flatten()
        if scale:
            pcf_four = self.scale_snapshots( pcf_four)
        return pcf_four


    def scale_snapshots(self, pcf_four):
        """ 
        Scale the snapshots of the 2pcf in real space given different 
        saling schemes. Applies a minimum 0 mean scaling. Is equivalent
        to the scale_snapshots implemented for the parent method (after
        applying ifft) Choose between
        None: (or baseline) zero mean each snapshot
        'max1':   scale every corner to have
        'fscale': filter out the volume fraction by division
        Parameters:
        -----------
        pcf_four:   numpy 2d-array of shape (?, n_s)
                    sample wise arranged data of snapshots
        Returns:
        --------
        pcf_four:   numpy 2d-array of shape (?, n_s)
                    sample wise shifted (and scaled) snapshots
        """
        vol = pcf_four[0,:]**0.5
        pcf_four[0,:] = 0 #zero mean snapshot
        if self.scaling == 'max1': #every corner has value 1
            pcf_four = pcf_four /( vol -vol**2 )
        elif self.scaling == 'fscale': #take out the volume fraction
            pcf_four = pcf_four /( vol )
        return pcf_four

    
    def get_xi( self, pcf, n_xi=None):
        """
        compute the reduced coefficients given the current basis
        """
        if self.B.shape[0] == pcf.shape[0]:
            return self.B[self.corner_indexing,:n_xi].T @ pcf[self.corner_indexing]
        else: #basis is in reduced representation
            try: #given full snapshots
                return self.B.T[:n_xi] @ pcf[ self.corner_indexing]
            except: #assume snapshots have been truncated accordingly
                return self.B.T[:n_xi] @ pcf


    def initialize_tracking_variables( self):
        """
        see the parent method for reference
        additional tracking variable:
        N_corner - total number of nonzero values at the current enrichment step
        """
        super().initialize_tracking_variables()
        self.N_corner = []


    def update_tracking_variables( self, *args, **kwargs):
        """
        see the parent method for reference, simply tracks also the
        size in the Fourier truncation if that has been implemented
        """
        super().update_tracking_variables( *args, **kwargs)
        if self.corner_indexing != slice( None):
            nonzeros = round( np.sum( self.B[:,0].flatten() !=0) /4)**0.5
            self.N_corner.append( nonzeros)
