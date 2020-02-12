import matplotlib.pyplot as plt
from palette import *

import matplotlib.font_manager as fm

def linestyles( x, y, z, lw=4, ls='-', **kwargs):
    """
    Pre defined linecolors given in a list, for each entry in the list there is one color
    The corporate colors are given in this order:
    blue, lightblue, green, magenate, gray, red, yellow

    can be used by e.g. plt.plot( x,y, **ls[0] )
    The given input arguments modify the style for every line.

    Parameters. optional:
    ---------------------
    lw          float, default = 4
                specified linewidth of all lines
    ls          string, default = '-'
                specified linestyle of all lines
    **kwargs    dict
                additional linestyles specified for each line 
    To see the available options in kwargs and ls, see "help( plt.plot)" for reference

    Returns:
    --------
    linestyles:     list
                    list of specified linestyles of various colors
    """

    styles = []
    styles.append( dict( linewidth=lw, color=CDColor( 'uniSblue'),    linestyle=ls, **kwargs ) ) # dark blue
    styles.append( dict( linewidth=lw, color=CDColor( 'uniSlblue'),   linestyle=ls, **kwargs ) ) # light blue
    styles.append( dict( linewidth=lw, color=CDColor( 'uniSgreen'),   linestyle=ls, **kwargs ) ) # green
    styles.append( dict( linewidth=lw, color=CDColor( 'uniSmagenta'), linestyle=ls, **kwargs ) ) # magenta
    styles.append( dict( linewidth=lw, color=CDColor( 'uniSgray'),    linestyle=ls, **kwargs ) ) # gray
    styles.append( dict( linewidth=lw, color=CDColor( 'uniSred'),     linestyle=ls, **kwargs ) ) # red
    styles.append( dict( linewidth=lw, color=CDColor( 'uniSyellow'),  linestyle=ls, **kwargs ) ) # yellow
    return styles


def rc_default( fontsize=20, grid=True):
    """
    Gives some default parameters set for a nice plot layout
    This function is to be used in the main as "plt.rcParams.update( rc_default() )"
    
    Parameters, optional:
    ---------------------
    fontsize:       int, default:20
                    fontsize of all things in the plot
    grid:           bool, default: True
                    If a grid should be plotted per default in each plot

    Returns:
    --------
    default_params: dict
                    dictionary of set default parameters 
    """
    default_params = dict() 
    ## Font specification
    try:
        uni_font = [ font for font in fm.findSystemFonts() if ('UniversforUniS65Bd-Regular.ttf' in font ) ][0]
    except:
        print( 'specified font for Uni Stuttgart not found, continuing with default font' )
    default_params.update( { 'font.family':uni_font, 'font.size':fontsize} )

    ## Default Grid
    if grid:
        default_params.update( { 'axes.grid':True, 'grid.color':'#AAAAAA', 'grid.linestyle':':', 'grid.linewidth':1.2 } )

    ## Spacing between subplots
#     'figure.subplot.bottom': 0.11,
#     'figure.subplot.hspace': 0.2,
#     'figure.subplot.left': 0.125,
#     'figure.subplot.right': 0.9,
#     'figure.subplot.top': 0.88,
#     'figure.subplot.wspace': 0.2,
    ## MUSS ICH NOCH FEINTUNEN, sodass es immer hübsch ist
    return default_params



def subplots( n_row=1, n_col=1, figsize=None, **kwargs):
    """
    Shadows the "plt.subplots" function and adds default styles.  See help( plt.subplots) for reference
    This function yields optimal results for these subplot layouts: 1x1, 1x2 or 1x3
     If "figsize" is not set, the spacing between the plots is also adjusted, otherwise it takes matplotlib defaults (for 1x1-1x3 plots)
    For all other parameters, only the figsize and no spacing is set
    
    Parameters, optional:
    ---------------------
    n_row:      int, default: 1
                number of rows for the subplot 
    n_col:      int, default: 1
                number of columns for the subplot 
    figsize:    list like of floats, default: None
                Set the figsize of the subplots
    **kwargs:   keyworded arguments
                arguments which will be put into the plt.subplots(**kwargs) call 
    Returns:
    --------
    fig:        plt.figure object
                figure handle for plot
    axes:       plt.axes object or np.ndarray of axes object
                axes object for each subplot 
    """

    if figsize is None:
        if n_col == 1 and n_row == 1:
            figsize = [ 6/2.54, 5/2.54] ###matplotlib kann nur inches, deswegen muss man hier für CM teilen
        elif n_col == 2 and n_row == 1:
            figsize = [ 12/2.54, 5/2,54 ]
            #fig.subplots.adjust( wspace=xx) #TODO adjust the spacing between the plots
        elif n_col == 3 and n_row == 1:
            figsize = [ 15/2.54, 5/2,54 ]

        else:
            figsize = [ 6*n_col/2.54, 5*n_row/2.54 ]
            if n_row > 3 or n_col > 3:
                print( '############### WARNING ###############\nfigsize set by default to {}, for more than 3 rows/columns in the plot could be ugly'.format( figsize) )

    fig, axes = plt.subplots( n_row, n_col, figsize=figsize, **kwargs)
    return fig, axes


def set_titles( axes, *titles):
    """
    Set the title of multiple axes objects in one function call

    Parameters:
    -----------
    axes:       plt.axes object or np.ndarray of axes objects
                axes handles on which the titles should be added
    *titles:    strings
                multiple strings for the titles, should be as many titles as there are axes handles given

    Returns:
    --------
    axes:       plt.axes object or np.ndarray of axes objects
                axes handles with set titles 
    """

    if isinstance( axes, np.ndarray):
        axes_shape = axes.shape
        if np.sum( axes.shape) != len( titles):
            print( "################### WARNING #####################\nmismatching number of titles and axes objects given, matching the first 'n axes' with first 'n titles' " )
        axes = axes.flatten()
        for i in range( np.min( (len( titles), len(axes) )) ):
            axes[i].set_title( titles[i] )
        axes = axes.reshape( axes_shape)
    else:
        try:
            axes.set_title( titles)
        except:
            print( "################### WARNING #####################\n title for single axes object could not be set, returning axes with no title added" )
    return axes



def add_legend( ax, position='top_right', opacity=0.5, **kwargs):
    """
    adds a legend to the handed axes object. default position of it is top right, the center is denoted by positon=(0.5, 0.5), the relative position with a tuple/list contianing floats is pretty striaght forward
    """
    defaults = dict( loc='center', bbox_to_anchor=(0.15, 0.9), fontsize=15, facecolor=uniSgray(), edgecolor=uniSblue() )
    if position==tuple or position==list:
        defaults['bbox_to_anchor']= position
    elif position==str:
        if position == 'top_left':
            relative_position=(0.15,0.85)
        if position == 'bot_left':
            relative_position=(0.15,0.15)
        elif position == 'top_right':
            relative_position=(0.85,0.85)
        elif position == 'bot right':
            relative_position=(0.85,0.15)
        elif position == 'center_right':
            relative_position=(0.85,0.5)
        elif position == 'bot_mid':
            relative_position=(0.5,0.85)
        defaults['bbox_to_anchor']=relative_position
    style= {**defaults, **kwargs} #overwrites the defaults by the kwargs
    key = ax.legend(**style)
    key.get_frame().set_linewidth( 5) 
    if 'linewidth' in kwargs or 'lw' in kwargs:
        try: 
            key.get_frame().set_linewidth( kwargs['linewidth']) 
        except:
            key.get_frame().set_linewidth( kwargs['lw']) 
        finally:
            print( 'wrong format of linewidth specified, returns to the default')

    return ax


def axis_labels( ax, xlabel, ylabel, fontsize=16, **kwargs):
    """
    Set the x and y label for a given ax object
    """
    ax.set_xlabel(xlabel, size=fontsize, **kwargs)
    ax.set_ylabel(ylabel, size=fontsize, **kwargs)
    return ax


def bounding_lines(ax, horizontal=True, minval=0, maxval=1):
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


def set_grid( ax, lw=2.5, ls=':', color='#AAAAAA'):
    ## DEPECRATED (is given in the default params function )
    """
    Set a grid to the given axes object
    
    Paremters:
    ----------
    ax:     plt.axes object
            current axes object to add the grid in

    lw:     float, default: 2.5
            linewidth of the grid
    ls:     string, defaiult: ':' 
            linestyle of the grid
    color:  string, default: '#AAAAAA'
            color of the grid

    Returns:
    --------
    ax:     plt.axes object
            axes object with added grid
    """
    ax.grid( linewidth=2.5, color='#AAAAAA', linestyle=':' )
    return ax
