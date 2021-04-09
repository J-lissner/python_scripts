import dill as pickle
import os
#import pickle

from importlib import import_module #as import_module
from zipfile import ZipFile

"""
Disclaimer - this seems very buggy with the sypder editor 
(because spyder has troubles reloading hand written objects (memory is flooded for some reason))
( This is due to the fact of interactive debugging - where reloading and "version check" of active modules slows it down alot)
To circumvent the spyder problem, launch python from the terminal
"""

# right now i just created some objects
# it might be better to just have functions, i will judge at the end
class Saver():
    """
    This Saver object is written on linux for linux
    All neccesary objects are dumped into the specified solver, and the Loader()
    object below reassembles the model for further usage.
    The Retrain() object is intended to load and save the model again 
    """
    def __init__(self, savepath, name_of_code, model_name, model, hard_overwrite=True, script_path=None):
        """
        Specify on where the model should be stored 
        as well as necessary information for reconstruction
        Depending on the user settings for the model,
        further methods need to be called
        Parameters:
        -----------
        savepath:       string
                        Path to store location,
                        has to match the path of the tf.train.CheckpointManager
        name_of_code:   string
                        name of the codefile where the model-object is defined
        model_name:     string
                        name of the model object in the codefile
        model:          tensorflow.keras.model
                        trained neural network 
        hard_overwrite: bool, default True
                        deletes the folder if it already exists
        script_path:    string, default None
                        path on where to look for default scripts used in main
                        Defaults to the script path of me (lissner)
        Returns:
        --------
        None:           stores files in the specified location
        """
        if script_path is None:
            self.script_path = '/home/lissner/scripts/python/tensorflow_functions/'
        else: 
            self.script_path = script_path
            if script_path[-1] != '/':
                self.script_path = self.script_path + '/'
        self.savepath = savepath

        if name_of_code[-3:] != '.py':
            name_of_code += '.py'
        if hard_overwrite is True:
            os.system( 'rm -r {}'.format( self.savepath) )
        os.system( 'mkdir -p {}'.format( self.savepath) )
        os.system( 'touch {}/__init__.py'.format( self.savepath) )
        if os.path.isfile( name_of_code): 
            os.system( 'cp {} {}/custom_model.py'.format( name_of_code, self.savepath) )
        else:
            print( 'file for models not found, taking default path of scripts:', self.script_path)
            os.system( 'cp {} {}/custom_model.py'.format( self.script_path+name_of_code, self.savepath) ) 

        with open( '{}/model_name.pkl'.format( self.savepath), 'wb') as pklfile:
            pickle.dump( model_name, pklfile) 
        model.save_weights('{}/weights/'.format( self.savepath), save_format='tf')


    def inputs(self, *args, **kwargs):
        """
        Save all the inputs of MyModel( *args, **kwargs) 
        The specified inputs are used on call for the ANN reconstruction
        Only the __init__ function is called, specific models need another function TODO
        (you should just copy ALL arguments of the initialization of the model in this function call)
        Parameters:
        -----------
        *args:      any number of unspecified arguments
                    all of the unspecified (required and optional) arguments 
                    used for calling the __init__ method
                    in correct order!
        **kwargs:   any number of keyworded arguments
                    all of the keyworded arguments used for calling the __init__ method
        Returns:
        --------
        None:       dumps the required 'metadata' into the specfied path
        """
        if args:
            with open( '{}/init_args.pkl'.format( self.savepath), 'wb') as pklfile:
                pickle.dump( args, pklfile)
        else:
            print( 'no args given, contuinuing but not saving any')

        if kwargs:
            with open( '{}/init_kwargs.pkl'.format( self.savepath), 'wb') as pklfile:
                pickle.dump( kwargs, pklfile)
        else:
            print( 'no kwargs given, contuinuing but not saving any')


    def locals(self, **kwargs):
        """
        Store local variables which are required for post-processing/retraining
        and have been defined in the __main__ file of training
        This could be e.g. the loss, optimizer, etc.
        Should always be used as 'locals( optimizer=optimizer, loss=loss, metric=metric)' etc.
        Parameters:
        -----------
        **kwargs:   key-value pairs of local variables in __main__
                    locally defined variables should be stored as
                    variable_descriptor=variable, e.g. loss=loss, etc.
        Returns:
        --------
        None:       Dumps the value of the variable into the specified folder
                    A tutorial on how to recover these values is given in the
                    official documentation of the 'load_locals()' method
        """
        if kwargs:
            with open( '{}/local_kwargs.pkl'.format( self.savepath), 'wb') as pklfile:
                pickle.dump( kwargs, pklfile)
        else:
            print( 'no kwargs given, please specify at least optimizer and loss' )


    def scaling(self, input_scaling, output_scaling):
        #TODO add another set of 'manual scaling' given as functions with inverse scaling requirement
        # then simply call these on the input when they are put LAST (to scale) on the data
        scalings = { 'input':input_scaling, 'output':output_scaling}
        with open( '{}/scalings.pkl'.format( self.savepath), 'wb') as pklfile:
            pickle.dump( scalings, pklfile)



    def tracked_variables(self, **kwargs):
        """
        This was a function to save some tracked variables in cooperation with the VISIS 
        @munz
        Can be used to just dump some (preferably lists or numpy arays)
        """
        # THis might be better with hdf5, depends on how large the data is gonna be
        # right now pickling some dictionaries seems really conveniant
        with open( '{}/tracked_variables.pkl'.format( self.savepath), 'wb') as pklfile:
            pickle.dump( kwargs, pklfile)


    def code( self, *args):
        """
        Save any code which is required for imports for the model/training
        The code has to be specifiied by the full path, if it does not exist
        the default script_path specified in __init__ will be used
        Parameters:
        -----------
        *args:      any number of strings
                    full path to the scripts which are required to 
                    reconstruct the model on loading
        Returns:
        --------
        None:       dumps a zip archive into the specified folder 
        """
        codefile = ZipFile( '{}/appended_code.zip'.format( self.savepath), 'w' )
        for code in args:
            if code[-3:] != '.py':
                code += '.py'
            if os.path.isfile( code):
                codefile.write( code )
            elif os.path.isfile( self.script_path + code):
                print( 'file was not found, read from default path of scripts:', self.script_path )
                codefile.write( self.script_path + code )
            else:
                print( 'WARNING: Following script to save has not been found: {}\n , might lead to issues on trying to reload the model'.format( code) )
        codefile.close()


class Loader():
    """
    Documentation very similar to the saver object, is omitted for now
    """
    def __init__(self, savepath):
        """ 
        Initialize the Loader pointing to the direction where the model is stored.
        Note that the Saver has to be correctly executed for the loader to work.
        This must include 'Saver.code()'. All other saver functions are conditional
        Parameters:
        -----------
        savepath:   string
                    path to the folder where the Saver() dumped the objects
        Returns:
        --------
        Loader:     instance of Loader object 
                    Loader object to assemble the previously stored model
        """
        self.path = savepath


    def model(self):
        """
        Return the model how it was saved. This accesses the saved 'Saver.code()' and
        (optional) the 'Saver.inputs()'
        Parameters:
        -----------
        None:       Already knows how to reassemble based on the Saver relation
        Returns:
        --------
        Model:      restored instance of the user defined model
                    restores the saved weights as well as the defined architecture
        """
        with open( '{}/model_name.pkl'.format( self.path), 'rb') as string_file:
            model_name = pickle.load( string_file)
        model_code = '{}/custom_model'.format( self.path).replace('/','.') 
        model_code = getattr( import_module( model_code), model_name) 

        try:
            with open( '{}/init_args.pkl'.format( self.path), 'rb') as pklfile:
                args = pickle.load( pklfile)
        except: 
            args = []
            print( 'no "init_args" found in the saved folder, contuniong without loading any') 
        try:
            with open( '{}/init_kwargs.pkl'.format( self.path), 'rb') as pklfile:
                kwargs = pickle.load( pklfile)
        except: 
            kwargs = {}
            print( 'no "init_kwargs" found in the saved folder, contuniong without loading any') 
        model = model_code( *args, **kwargs)
        model.load_weights( '{}/weights/'.format( self.path) )
        return model


    def locals(self, tutorial=True):
        """
        Restore the locally defined variables in the previous training.
        This function requires additional code in the main file to work (see below)
        If the code is executed, the local variables are usable 
        as defined in the previous training.  
        how to restore the local variables:
        for key, value in Loader.locals( tutorial=False).items(): 
            exec(key + '=value')\n
        This will automatically assign all the previously saved variables\n
        Parameters:
        -----------
        tutorial:   bool, default True
                    If True, how to correctly restore the variables is printed
                    to console
        Returns:
        --------
        local_variables:    dict
                            dictionary containing the 'variable'-'variable_value' pair
        """ 
        with open( '{}/local_kwargs.pkl'.format( self.path), 'rb') as pklfile:
            local_variables = pickle.load( pklfile)
        if tutorial:
            print( """###########################################################################
               After the variables have been loaded they need to be assigned by
               for key, value in Loader.locals( tutorial=False).items(): 
                   exec(key + '=value')\n
               This will automatically assign all the previously saved variables\n
               """)
            print( '###########################################################################' )
        print()
        print( 'additional variables when loading the locals:', list( local_variables.keys() ) )
        print( 'NOTE: existing variables will be overwritten!')
        print()
        return local_variables


    def scaling(self ):
        """
        Load the previously stored scalings for the data.
        The scalings should be computed with the module 'data_processing' 
        (the module is in the same git repository)
        Parameters:
        -----------
        None:   The Loader refers to the Saver relation
        Returns:
        --------
        scalings:   Dict
                    Input and output scalings stored under the
                    'input' and 'output' keys 
        """
        with open( '{}/scalings.pkl'.format( self.path), 'rb') as pklfile:
            scalings = pickle.load( pklfile)
        return scalings


    def tracked_variables(self ):
        """
        Load additionally tracked variables after training
        Parameters:
        -----------
        None:   The Loader refers to the Saver relation
        Returns:
        --------
        tracked:    Dict
                    Dictionary containing all the tracked variables
        """
        with open( '{}/tracked_variables.pkl'.format( self.path), 'rb' ) as pklfile:
            track = pickle.load( pklfile)
        return track



class Retraining():
    """
    Load a previously trained model and then save it into the specified path
    The save path can be the same as the load path, then the previous model is overwritten
    """
    def __init__( self, load_path, save_path):
        self.load_path = load_path 
        self.save_path = save_path 

        with open( '{}/local_kwargs.pkl'.format( self.load_path), 'rb') as pklfile:
            local_variables = pickle.load( pklfile)
        self.local_variables = local_variables


    def load_model(self ):
        with open( '{}/model_name.pkl'.format( self.load_path), 'rb') as string_file:
            model_name = pickle.load( string_file)
        model_code = '{}/custom_model'.format( self.load_path).replace('/','.') 
        model_code = getattr( import_module( model_code), model_name)

        try:
            with open( '{}/init_args.pkl'.format( self.load_path), 'rb') as pklfile:
                args = pickle.load( pklfile)
        except: 
            args = []
            print( 'no "init_args" found in the saved folder, contuniong without loading any')

        try:
            with open( '{}/init_kwargs.pkl'.format( self.load_path), 'rb') as pklfile:
                kwargs = pickle.load( pklfile)
        except: 
            kwargs = {}
            print( 'no "init_kwargs" found in the saved folder, contuniong without loading any')

        model = model_code( *args, **kwargs)
        model.load_weights( '{}/weights/'.format( self.load_path) )
        self.model_inputs = [ args, kwargs]
        return model


    def load_scaling(self ):
        with open( '{}/scalings.pkl'.format( self.load_path), 'rb') as pklfile:
            scalings = pickle.load( pklfile)
        return scalings

    def load_locals(self, tutorial=True):
        with open( '{}/local_kwargs.pkl'.format( self.load_path), 'rb') as pklfile:
            local_variables = pickle.load( pklfile)
        if tutorial:
            print( """###########################################################################
               After the variables have been loaded they need to be assigned by
               for key, value in Loader.locals( tutorial=False).items(): 
                   exec(key + '=value')\n
               This will automatically assign all the previously saved variables\n
               """)
            print( '###########################################################################' )
        print()
        print( 'additional variables when loading the locals:', list( local_variables.keys() ) )
        print( 'NOTE: existing variables will be overwritten!')
        print()
        return local_variables


    def save_model( self, model):
        os.system( 'mkdir -p {}'.format( self.save_path) ) #simply does nothing if it already exists
        model.save_weights('{}/weights/'.format( self.save_path), save_format='tf')
        if self.save_path == self.load_path:
            return

        ## else copy everything to the new folder
        os.system( 'touch {}/__init__.py'.format( self.save_path) )
        os.system( 'cp {}/custom_model.py {}/custom_model.py'.format( self.load_path, self.save_path) ) 
        os.system( 'cp {}/model_name.pkl  {}/model_name.pkl'.format( self.load_path, self.save_path) ) 
        os.system( 'cp {}/init_args.pkl   {}/init_args.pkl'.format( self.load_path, self.save_path) ) 
        os.system( 'cp {}/init_kwargs.pkl {}/init_kwargs.pkl'.format( self.load_path, self.save_path) ) 

        ## copy previously stored locals and scalings
        self.save_locals()
        self.save_scaling()


    def save_locals(self, **kwargs):
        """
        saves the local variables defined in the __main__ file
        local variables refers to definitions of the ANN, optimizer, loss, etc.
        (where the ANN was trained)
        If there are no locals given, then the previously stored locals are copied 
        """
        self.local_variables = { **self.local_variables, **kwargs }
        with open( '{}/local_kwargs.pkl'.format( self.save_path), 'wb') as pklfile:
            pickle.dump( self.local_variables, pklfile)


    def save_scaling(self, input_scaling=None, output_scaling=None):
        if input_scaling is None and output_scaling is None:
            os.system( 'cp {}/scalings.pkl   {}/scalings.pkl'.format( self.load_path, self.save_path) ) 
            return
        else:
            scalings = { 'input':input_scaling, 'output':output_scaling}
            with open( '{}/scalings.pkl'.format( self.save_path), 'wb') as pklfile:
                pickle.dump( scalings, pklfile)

