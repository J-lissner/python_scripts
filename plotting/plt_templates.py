import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
import matplotlib
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle
## personal packages
from palette import UniStuttgart as uniS

print( '"import plt_templates..." Default parameters for matplotlib.pyplot ",\
      have to be updated, use "plt.rcParams.update( plt_templates.rc_default())"')

def linestyles( *args, **kwargs):
    """
    Pre defined linecolors given in a list, for each entry in the list there is one color
    The corporate colors are given in this order:
    blue, lightblue, green, magenate, gray, red, yellow 
    can be used by e.g. plt.plot( x,y, **ls[0] )
    The given input arguments (args, kwargs) modify the style for every line.
    The input arguments can be interpreted as plt.plot( x, y, *args, **kwargs) 
    Parameters. optional:
    ---------------------
    *args:      unpacked list
                additional linestyles specified for each line via args
    **kwargs:   unpacked dict
                additional linestyles specified for each line via kwargs
    To see the available options see "help( plt.plot)" for reference 
    Returns:
    --------
    linestyles:     list
                    list of specified linestyles of various colors
    """ 
    styles = []
    styles.append( dict( *args, color=uniS.blue,    **kwargs ) ) # dark blue
    styles.append( dict( *args, color=uniS.lblue,   **kwargs ) ) # light blue
    styles.append( dict( *args, color=uniS.green,   **kwargs ) ) # green
    styles.append( dict( *args, color=uniS.magenta, **kwargs ) ) # magenta
    styles.append( dict( *args, color=uniS.gray,    **kwargs ) ) # gray
    styles.append( dict( *args, color=uniS.red,     **kwargs ) ) # red
    styles.append( dict( *args, color=uniS.yellow,  **kwargs ) ) # yellow
    return styles


def rc_default( fontsize=11.7, ticksize=9, legend_fontsize=10.2, \
               axis_label_size=None, use_tex=True, grid=True, **kwargs):
    """
    Gives some default parameters set for a nice plot layout
    This function is to be used in the main as "plt.rcParams.update( rc_default() )"
    
    Parameters, optional:
    ---------------------
    fontsize:       float, default:20
                    fontsize of all things in the plot
    grid:           bool, default: True
                    If a grid should be plotted per default in each plot
    ticksize:       float, default None
                    fontsize of the tikz, is set to 0.6*'fontsize' if not specified 
    **kwargs:       unpacked dict
                    hard overwrites the defaults with any given rcParams (in kwargs). 
                    If no kwargs are given, the defaults are set 
    Returns:
    --------
    default_params: dict
                    dictionary of set default parameters 
    """
    default_params = dict() 
    ## Font and text specification
    try:
        uni_font = [ font for font in fm.findSystemFonts() if ('UniversforUniS65Bd-Regular.ttf' in font ) ][0]
        default_params.update( { 'font.family':uni_font, 'text.usetex':use_tex } )
    except:
        print( 'Uni stuttgart font not installed, continuing with default font' )
    default_params.update( { 'font.size':fontsize} )
    default_params.update( { 'xtick.labelsize':ticksize, 'ytick.labelsize':ticksize } )

    ## layout of the figure, sizes
    default_params.update( {'axes.linewidth': 1.3 } )
    default_params.update( {'axes.titlepad':3} ) 
    if( axis_label_size):
        default_params.update( {'axes.labelsize':axis_label_size} ) 
    default_params.update( {'xtick.major.pad':1.5, 'ytick.major.pad':1.5 } )
    default_params.update( {'xtick.major.size':2.5, 'ytick.major.size':2.5} )
    ## Default Grid
    if grid:
        default_params.update( { 'axes.grid':True, 'grid.color':'#AAAAAA', 'grid.linestyle':':', 'grid.linewidth':0.8 } )
    ## legend
    default_params.update( {'legend.fontsize': legend_fontsize } )
    ## linewidth and scatter style
    default_params.update( {'lines.linewidth': 2, 'lines.markeredgecolor': 'black' } )
    default_params.update( { 'scatter.edgecolors': 'black', 'lines.markersize':4 } )
    ## default color palette to be uniStuttgart colors (cycles for line AND scatterplot
    default_params.update( {'axes.prop_cycle': plt.cycler('color', uniS.colorwheel) } ) 
    default_params.update( **kwargs)  #overwrite any setting by the specified kwargs
    ### default latex packages
    default_params.update( {'text.latex.preamble': r'\usepackage{amsfonts,amsmath,amssymb}' } )
    return default_params



def fixed_plot( n_row=1, n_column=1, x_stretch=1, y_stretch=1, padding=[0,0], **kwargs):
    """
    Return matplotlib fig, axes instance for single plot. 
    Adjusted sizes (font etc) have to be set in the plt.rcParams, not locally on the axes object
      e.g. with the "rc_default" function in this module
    The plot of the exported figure with fig.savefig() is exactly of size  6x5 cm. DO NOTE USE THE OPTION "bbox_inches" 
    If the width/height of the plot is to be adjusted, use x_stretch or "ystretch"
    The resulting figure size will be printed in terminal (in centimeteres)
    EXAMPLE:
    --------
    fig, ax = default_plot( y_stretch = 6/5)
    will return a plot of size 6x6 (default x=6, y=5 * 6/5) (figsize is larger, printed to stdout)
    
    Parameters:
    -----------
    n_row:          int, default 1
                    how many rows of subplots should be returned
    n_column:       int, default 1
                    how many columns of subplots should be returned
    x_stretch:      float, default 1
                    stretch of each plot in x direction (size adjustment)
    y_stretch:      float, default 1
                    stretch of each plot in y direction (size adjustment)
    padding:        list of 2 floats:
                    padding to the left and top of the subplots
                    NOTE: this parameter is buggy and rescales the subplots
                        but i will leave it for now and use it for personal purposes
    **kwargs:       dict
                    input kwargs for plt.subplots( **kwargs), NO GUARANTEES MADE (YET) 
    Returns:
    --------
    fig:            matplotlib.pyplot figure object
                    figure handle for the specified plot
    axes:           matplotlib.pyplot axes object
                    axes handle for the specified plot 
    """
    ## default parameters and hardwirde constants
    cm_conversion = 2.3824 #factor that the specified width/height is given in cm 
    x_pad = 0.03 #flaoting space right of the subplot
    y_pad = 0.105 #floating space at the top of the subplot (space for title basically)
    x_offset = 0.2 
    y_offset = 0.145


    ## space adjustment for different fontsizes
    default_labelsize = 9 #NOTE HARD WIRED AS A REFERENCE (DOES NOT HAVE TO MATCH "plt_templates" OR YOUR LOCAL "plt.rcParams"
    default_fontsize = 11.7
    #considering the ticksize
    try:
        x_offset += (plt.rcParams['xtick.labelsize']/ default_labelsize -1) * 0.053  #yticks
    except:
        x_offset += 0.053 #yticks, default value
    try:
        y_offset += (plt.rcParams['xtick.labelsize']/ default_labelsize -1) * 0.030  #xticks
    except: 
        y_offset += 0.030  #xticks, default value
    # consideration of labels and title
    x_offset += (plt.rcParams['font.size']/ default_fontsize-1 )* 0.035 #ylabel
    y_offset += (plt.rcParams['font.size']/ default_fontsize-1 )* 0.030 #xlabel
    y_offset += (plt.rcParams['font.size']/ default_fontsize-1 )* 0.047 #title
    x_offset = x_offset /n_column /x_stretch  
    y_offset = y_offset /n_row /y_stretch    

    ## size of the ax and figure
    default_axwidth = 0.97 /n_column  
    default_axheight = 0.895 / n_row
    required_axwidth =  default_axwidth - x_offset 
    required_axheight = default_axheight - y_offset
    additional_width = default_axwidth/required_axwidth
    additional_height = default_axheight/required_axheight
    required_width =   6 /default_axwidth   *x_stretch *additional_width  / cm_conversion  
    required_height =  5 /default_axheight  *y_stretch *additional_height / cm_conversion
    required_width += padding[0]
    required_height += padding[1]
    padding_offset = [padding[0]/required_width, padding[1]/required_height]
    y_extra = -padding_offset[1]/n_row
    x_extra = 0#padding_offset[0]/n_column
    ax_position = np.zeros( (n_row, n_column), dtype=object )
    for i in range( n_row):
      for j in range( n_column):
        ax_position[i,j] =  [ x_offset + ( default_axwidth + x_pad/n_column + x_extra )*j,
                    y_offset +(default_axheight+y_pad/n_row + y_extra)*i, required_axwidth, required_axheight ]  

    ## setting of figure and axes object
    fig, axes = plt.subplots( n_row,n_column, **kwargs)
    fig.savefig = savefig( fig )
    fig.canvas.draw()
    fig.set_constrained_layout(False)
    fig.set_size_inches( required_width, required_height)

    if n_column == 1 and n_row == 1:
        axes = np.array( [[axes]] )
    elif n_column == 1 and n_row !=1:
        axes = axes[:,None]
    elif n_column != 1 and n_row ==1:
        axes = axes[None,:]
    for i in range( n_row):
        for j in range( n_column):
            axes[i,j].set_position( ax_position[-(i+1),j] )
            axes[i,j].set( xlabel='x', ylabel='f(x)' )
    print( 'Size of the full figure:', round(required_width*cm_conversion,3), '/', round( required_height*cm_conversion, 3), f'[cm]; {round( required_width,3)}/{round( required_height,3)} [in]' )
    if n_column == 1 and n_row == 1:
        return fig, axes[0][0]
    else:
        return fig, axes.squeeze()


#### LEGEND RELATED FUNCTIONS ####
class HandlerColormap(HandlerBase):
    """
    Create some handle which can be used for plotting inside a legend
    Does have to be paired with a dummy mappable and the label is required
    to be added manually.
    It is advised to pair the 'cmap' parameter in the init with a custom
    colormap
    All of this is obtained by the 'get_legend_entries' method, which should
    be used as ax.legend( **HandlerColormap.get_legend_entries() )
    ######### Example: ##########
    from matplotlib.colors import LinearSegmentedColormap as Cmap
    colors = [uniS.gray60, uniS.gray20, uniS.gray]
    colormap = Cmap.from_list( "", colors ) ]
    custom_label = plt_templates.HandlerColormap( colormap, label='text', n_stripes=len( colors))
    plt_templates.add_legend( axes_object, **custom_label.get_legend_entries( axes_object) )
    ## the axes_object ensures that previously added labels will not be discarded
    ## read the further documentation to find more flexibility in the legend design
    """
    def __init__(self, cmap, label, n_stripes=3, **kwargs):
        """
        Parameters:
        -----------
        cmap:       matplotlib.pyplot.cm colormap
                    colormap to plot in the legend
        label:      string
                    label to add to the colormap
        n_stripes:  int, default 3
                    in how many discrete colors the colormap should be split
        **kwargs:   kwargs directly passed to the super init
        """
        super().__init__(**kwargs)
        self.label = label
        self.cmap = cmap
        self.num_stripes = n_stripes

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        """ required function for matplotlib internal call """
        stripes = []
        for i in range(self.num_stripes):
            s = Rectangle([xdescent + i * width / self.num_stripes, ydescent],
                          width / self.num_stripes,
                          height,
                          fc=self.cmap((2 * i + 1) / (2 * self.num_stripes)),
                          transform=trans)
            stripes.append(s)
        return stripes

    def get_legend_entries( self, ax=None, other_handlers=[]):
        """
        If ax is given the existing legend handles of the axes object
        are also plotted.
        Parameters:
        -----------
        ax:             matplotlib.pyplot.axes object
                        axes object where the data was plotted
        other_handlers: list of HandlerColormap
                        other colormap handles to add to the legend
        Returns:
        --------
        legend_kwargs:  dict
                        dict to put into the ax.legend() call as kwargs
                        Contains all the required handles and labels
        """
        labels  = [self.label] + [x.label for x in other_handlers]
        handles = [Rectangle(( 0,0), 1, 1) for _ in range( 1+len(other_handlers) )]
        if ax is not None:
            additional_labels = ax.get_legend_handles_labels()
            labels += additional_labels[1]
            handles += additional_labels[0]
        handler_map = dict( zip( handles, [self]+other_handlers ))
        return dict( labels=labels, handles=handles, handler_map=handler_map )




def add_legend( ax, *legend_args, position='top right', opacity=0.8, **kwargs):
    """
    Add a legend to the specified ax object. Kwargs do overwrite parameters in plt.rcParams
    Parameters:
    -----------
    ax:         plt.axes object
                axes in which the legend should be added 
    position:   string, default 'top right'
                set the legend in a specified corner ( 'top/upper or bot/lower left/right', has a space in the string
                A dictionary is also accepted, should contain the keys to specify the position
                (prefered is 'loc' and 'bbox_to_anchor')
    opacity.    float, default 0.8
                opacity of the legend box
    **kwargs    unpacked dictionary
                additional parameters to customize the legend, e.g.
                linewidth or lw
                handlelength, handletextpad, labelspacing
                edgecolor, facecolor, fancybox, shadow
    """
    if not ax.get_legend_handles_labels()[1] and not kwargs:
        print( 'did not find any legend labels in plt_templates.add_legend(), returning...' )
        return #if there is nothing to put in the legend
    defaults =  dict( handlelength=1.8, handletextpad=0.4, labelspacing=0.5, 
                      fancybox=False, #shadow=True,  #shadow and opacity dont mix well
                      edgecolor=uniS.blue, facecolor=uniS.gray20, framealpha=opacity ) 
    if isinstance( position, str):
        position = position.replace( '_', ' ' )
        if position == 'bot left' or position=='lower left':
            defaults.update( dict( loc='lower left', bbox_to_anchor=(-0.01,-0.010) ) )
        elif position == 'top left' or position=='upper left':
            defaults.update( dict( loc='upper left', bbox_to_anchor=(-0.01,1.015) ) )
        elif position == 'top right' or position=='upper right':
            defaults.update( dict( loc='upper right', bbox_to_anchor=(1.005, 1.015) )  )
        elif position == 'bot right' or position=='lower right':
            defaults.update( dict( loc='lower right', bbox_to_anchor=(1.005, -0.010) )  )
    elif isinstance( position, dict):
        defaults.update( position)
    else:
        print( "Non allowed argument for 'position' in 'add_legend', setting default (top right) ")
        defaults.update( dict( loc='lower left', bbox_to_anchor=(-0.01,-0.010) ) )

    style= {**defaults, **kwargs} #overwrites the defaults by the kwargs
    legend = ax.legend( *legend_args, **style)
    legend.get_frame().set_linewidth( 1.0) 
    if 'linewidth' in kwargs or 'lw' in kwargs:
        try: 
            legend.get_frame().set_linewidth( kwargs['linewidth']) 
        except:
            legend.get_frame().set_linewidth( kwargs['lw']) 
        finally:
            print( 'wrong format of linewidth specified, returns to the default') 
    return legend


def export_legend( legend, savefile='./legend.pdf', expand=[-2,-2,2,2]):
    """
    export a standalone legend which didn't fit on the original image
    The legend should be extracted by firstly plotting the legend and then 
    passing it to this function., e.g.
    Example call:
    axes[1].grid( 'off') #removes background in opacity
    legend = plot.add_legend( axes[1] )
    plot.export_legend( legend, savepath + legend_savename)
    legend.remove() 
    axes[1].grid()
    Parameters:
    -----------
    legend:         matplotlib.legend.Legend object, or ax object
                    object which contains either a predefined legend, or ax
                    which will generate the legend using "add_legend function"
    savefile:       str, default './legend.pdf'
                    path and name of file to save to
    expand:         list of 4 floats
                    i think it has something todo with the size
    Returns:
    --------
    None:           only saves the legend to savefile 
    """
    temp_legend = False
    if not isinstance( legend, matplotlib.legend.Legend):
        ax = legend
        ylim = ax.get_ylim()[1]
        xlim = ax.get_xlim()[1]
        ax.set_ylim( ymax=1e10 )
        ax.set_xlim( xmax=1e10 )
        ax.grid( False)
        legend = add_legend( ax)
        temp_legend = True 
    ## get the figure for the legend
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents( *( bbox.extents + np.array(expand)))
    bbox = bbox.transformed( fig.dpi_scale_trans.inverted()) 
    fig.savefig(savefile, dpi="figure", bbox_inches=bbox) 
    if temp_legend:
        ax.grid()
        legend.remove()
        ax.set_ylim( ymax=ylim)
        ax.set_xlim( xmax=xlim)

def savefig( fig, *args, **kwargs):
    """
    emulated wrapper function to extend the default behaviour of 
    fig.savefig by printing out the path it saves to. To add this 
    functionality do:
    fig.savefig = plt_templates.savefig( fig)
    """
    _savefig = fig.savefig
    def wrapper( *args, **kwargs):
        print( f'saving figure to: {args[0]}' ) 
        return _savefig( *args, **kwargs)
    return wrapper



def export_borderless( image, savename, file_ending='.pdf', cmap=None, size=None, clim=None, cbar=False):
    """
    Export a single matrix (from imshow) as borderless image
    and save it to <savename+file_ending>. Intended to be used for RVEs
    Parameters:
    -----------
    image:      numpy 2d-array
                image data which should be plotted
    savename:   str
                path to file, if file ending is given here the next parameter
                will be ignored
    file_ending:str, default '.pdf'
                file format of the plot
    cmap:       str, default None
                if the cmap is not specified, default to gray with a custom colorrange
    size:       tuple of 2 ints, default None
                size of the figure in inches (in matplotlib)
    clim:       tuple of 2 ints, default None
                specified cmap range, always takes [0,1.2] if the colormap is 'gray'
    cbar:       bool, default False
                if the colorbar should be exported as well
    Returns:
    --------
    None:       saves to file
    """
    if size:
        fig, ax = plt.subplots( figsize=size)
    else: 
        fig, ax = plt.subplots()
    if not '.' in savename:
        savename = savename + file_ending
    if cmap is None:
        clim = [0, 1.2]
        cmap = 'gray' 
    img = ax.imshow( image, cmap=cmap)
    if clim is not None:
        img.set_clim( *clim )
    if cbar is True:
        plt.colorbar( img, ax=ax)
    ax.axis('off')
    fig.savefig( savename, bbox_inches='tight', pad_inches=0 )
    plt.close()
    return


#### OTHER DECORATING FUNCTIONS FOR CONVENIENCE ####
def set_titles( axes, *titles, **kwargs ):
    """
    Set the title of multiple axes objects in one function call
    Parameters:
    -----------
    axes:       plt.axes object or np.ndarray of axes objects
                axes handles on which the titles should be added
    *titles:    strings
                multiple strings for the titles, should be as many titles as there are axes handles given
    **kwargs:   dict
                formatting kwargs for the title                
    Returns:
    --------
    axes:       plt.axes object or np.ndarray of axes objects
                axes handles with set titles 
    """
    if isinstance( axes, np.ndarray):
        axes_shape = axes.shape
        if np.prod( axes.shape) != len( titles):
            print( "################### WARNING #####################\nmismatching number of titles and axes objects given, matching the first 'n axes' with first 'n titles' " )
        axes = axes.flatten()
        for i in range( np.min( (len( titles), len(axes) )) ):
            axes[i].set_title( titles[i], **kwargs )
        axes = axes.reshape( axes_shape)
    else:
        try:
            axes.set_title( *titles, **kwargs)
        except:
            print( "################### WARNING #####################\n title for single axes object could not be set, returning axes with no title added" )
    return axes



def bounding_lines(ax, horizontal=True, minval=0, maxval=1):
    ##(THIS FUNCTION IS PROLLY NOT REQUIRED)
    """ 
    adds a grey horizontal line at ymin and ymax
    input:  ax - axes object of current figure
            minvals [0,1] - value of first line to plot, default 0
            maxvals [0,1] - value of second line to plot, default 1
    returns: ax - axes object with additional lines
    """
    if horizontal:
        ax.axhline(minval, color='#AAAAAA', linewidth=3)
        ax.axhline(maxval, color='#AAAAAA', linewidth=3)
    else:
        ax.axvline(minval, color='#AAAAAA', linewidth=3)
        ax.axvline(maxval, color='#AAAAAA', linewidth=3)
    return ax

def layout(ax, title=None, titlesize=None, ticksize=None, gridstyle=None, \
           legendstyle=None, axislabelsize=None ) :
    """
    Sets a default layout around the given plot in "ax"
    Input: ax - axes object of current figure
           OPTIONAL ARGUMENTS
           title       - string; specified title to add
           tiksize     - int;    size of tikz of x and y
           gridstyle   - dict;   style of the grid
           legendstyle - dict;   style of the legend
    returns: ax - axes object with added layout
    """
    if gridstyle :
        ax.grid(**gridstyle)
    if legendstyle :
        ax.legend(**legendstyle)
    if ticksize :
        ax.tick_params( labelsize=ticksize)
    if axislabelsize :
        plt.rc('axes', labelsize=axislabelsize )
    if title :
        if titlesize:
            ax.set_title(title, fontsize=titlesize )
        else:
            ax.set_title(title)
    return ax


def imshow_with_colorbar( ax, image, OPT='ARGS'):
    return
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    gcf = ax.imshow( image)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar( gcf, cax=cax ) #oder so
    # TODO IMPLEMENT THIS FUNCTION


#### ALIAS/SHADOWS OF FUNCTIONS TO CATCH TYPOS ####
# alias of different functions
def rc_defaults( *args, **kwargs):
    """ see rc_default """
    return rc_default( *args, **kwargs)
def rcDefaults( *args, **kwargs):
    """ see rc_default """
    return rc_default( *args, **kwargs)
def rcDefault( *args, **kwargs):
    """ see rc_default """
    return rc_default( *args, **kwargs) 

def save_legend( *args, **kwargs):
    """ see export_legend """
    return export_legend( *args, **kwargs) 
def export_single( *args, **kwargs):
    """ see export_borderless """
    return export_borderless( *args, **kwargs)

