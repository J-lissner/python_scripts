import numpy as np
import matplotlib.pyplot as plt 
#import sys
#sys.path.append('plt_template/')
from palette import UniStuttgart as uniS

import matplotlib.font_manager as fm
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
    default_params.update( {'axes.prop_cycle': plt.cycler('color', [uniS.lblue, uniS.magenta, uniS.green, uniS.blue, uniS.gray30, uniS.red, uniS.yellow ]) } )

    default_params.update( **kwargs)  #overwrite any setting by the specified kwargs
    return default_params



def fixed_plot( n_row=1, n_column=1, x_stretch=1, y_stretch=1, **kwargs):
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
    **kwargs:       dict
                    input kwargs for plt.subplots( **kwargs), NO GUARANTEES MADE (YET)

    Returns:
    --------
    fig:            matplotlib.pyplot figure object
                    figure handle for the specified plot
    axes:           matplotlib.pyplot axes object
                    axes handle for the specified plot 
    """
    cm_conversion = 2.3824 #factor that the specified width/height is given in cm

    ## default spacings
    x_pad = 0.03 #flaoting space right of the subplot
    y_pad = 0.105 #floating space at the top of the subplot (space for title basically)
    x_offset = 0.2 
    y_offset = 0.145


    ## space adjustment for different fontsizes
    default_labelsize = 9 #NOTE HARD WIRED AS A REFERENCE (DOES NOT HAVE TO MATCH "plt_templates" OR YOUR LOCAL "plt.rcParams"
    default_fontsize = 11.7
    #considering the ticksize
    x_offset += (plt.rcParams['xtick.labelsize']/ default_labelsize -1) * 0.053  #yticks
    y_offset += (plt.rcParams['xtick.labelsize']/ default_labelsize -1) * 0.030  #xticks
    # consideration of labels and title
    x_offset += (plt.rcParams['font.size']/ default_fontsize-1 )* 0.035 #ylabel
    y_offset += (plt.rcParams['font.size']/ default_fontsize-1 )* 0.030 #xlabel
    y_offset += (plt.rcParams['font.size']/ default_fontsize-1 )* 0.047 #title
    x_offset = x_offset/n_column /x_stretch
    y_offset = y_offset/n_row /y_stretch

    ## size of the ax and figure
    default_axwidth = 0.97 /n_column  
    default_axheight = 0.895 / n_row
    required_axwidth =  default_axwidth - x_offset 
    required_axheight = default_axheight - y_offset
    additional_width = default_axwidth/required_axwidth
    additional_height = default_axheight/required_axheight
    required_width =   6 /default_axwidth   *x_stretch *additional_width  / cm_conversion
    required_height =  5 /default_axheight  *y_stretch *additional_height / cm_conversion

    ax_position = np.zeros( (n_row, n_column), dtype=object )
    for i in range( n_row):
        for j in range( n_column):
            ax_position[i,j] =  [ x_offset + ( default_axwidth + x_pad/n_column )*j, y_offset +(default_axheight+y_pad/n_row )*i, required_axwidth, required_axheight ]  

    ## setting of figure and axes object
    fig, axes = plt.subplots( n_row,n_column)
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
    print( 'Size of the full figure:', round(required_width*cm_conversion,3), 'x', round( required_height*cm_conversion, 3), '[cm]' )
    if n_column == 1 and n_row == 1:
        return fig, axes[0][0]
    else:
        return fig, axes.squeeze()


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



def add_legend( ax, position='top right', opacity=0.8, **kwargs):
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
    defaults =  dict( handlelength=1.8, handletextpad=0.4, labelspacing=0.5, 
                      fancybox=False, #shadow=True,  #shadow and opacity dont mix well
                      edgecolor=uniS.blue, facecolor=uniS.gray80, framealpha=opacity ) 
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
    key = ax.legend(**style)
    key.get_frame().set_linewidth( 1.0) 
    if 'linewidth' in kwargs or 'lw' in kwargs:
        try: 
            key.get_frame().set_linewidth( kwargs['linewidth']) 
        except:
            key.get_frame().set_linewidth( kwargs['lw']) 
        finally:
            print( 'wrong format of linewidth specified, returns to the default') 
    return ax


def axis_labels( ax, xlabel, ylabel, **kwargs):
    """
    Set the x and y label for a given ax object
    Parameters:
    -----------
    ax:         plt.axes object
                axes handle of the current plot
    xlabel:     string
                xlabel to set
    ylabel:     string
                ylabel to set
    **kwargs:   unpacked dict
                kwargs passed to ax.set_?label( **kwargs)
    Returns:
    --------
    ax          plt.axes object
                axes handle with the added labels
    """
    ax.set_xlabel(xlabel, **kwargs)
    ax.set_ylabel(ylabel, **kwargs)
    return ax





def export_legend( legend_handles, savefile='./legend.pdf'):
    """
    export a standalone legend which didn't fit on the original image
    The legend handles can be given by ax.get_legend_handles_labels()
    Parameters:
    -----------
    legend_handles: tuple of legend handles
                    found handles in the ax object
    savefile:       str, default './legend.pdf'
                    path and name of file to save to
    Returns:
    --------
    None:           only saves the legend to savefile

    """
    #fig_leg = plt.figure()
    #ax_leg = fig_leg.add_subplot(111)
    fig, ax = plt.subplots()
    ax.legend( *legend_handles, loc='center')
    ax.axis('off')
    fig.savefig( savefile, bbox_inches='tight')


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


#shadows of rc_default function to ommit typos
# alias of different functions
def rc_defaults( *args, **kwargs):
    return rc_default( *args, **kwargs)
def rcDefaults( *args, **kwargs):
    return rc_default( *args, **kwargs)
def rcDefault( *args, **kwargs):
    return rc_default( *args, **kwargs)

def save_legend( *args, **kwargs):
    return export_legend( *args, **kwargs) 
