import matplotlib.pyplot as plt
from palette import *

#NOTE TODO
#wir können hier auch weiter unten einige matplotlib funktionen aliasen, also default_figure als subplots
# oder add_legend als nur legend, dass eben des plt.legend dann mit template.legend ersetzt wird
#NOTES

def linestyles(linestyle='-', linewidth=3, mec='black', markersize=2, **kwargs):
    #kwargs macht es möglich dass man quasi alles normal übergibt, und dann kriegt man jede default linie mit ihrer farbe in dem gewünschten stil zurück zurück
    """
    Default linestyles in a list
    Easily accessed with e.g. ls=linestyles()... ax.plot(..., **ls[i] )
    Lines by color, here in the (help) elaboration: all the colors for the n lines are displayed
    """
    styles = []
    kwargs = { **{'mec':mec, 'markersize':markersize}, **kwargs} #put some defaults in the kwargs 
    #hier definiert man dann quasi die default farbe pro linie
    styles.append( dict( ls=linestyle, lw=linewidth, color=uniSblue(), **kwargs) )
    styles.append( dict( ls=linestyle, lw=linewidth, color=uniSred(), **kwargs) )
    # etc...
    return styles

def default_figure( n_row=1, n_col=1, figsize=(6,5), other_default='gotta think of them', **kwargs):
    fig, axes = plt.subplots( n_row, n_col, figsize=figsize, **kwargs)
    if axes.ndim > 1:
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                axes[i,j] = default_layout( axes[i,j], **kwargs)
    else:
        for i in range(axes.shape[0]):
            axes[i] = default_layout( axes[i], **kwargs)
    return fig, axes


def default_layout( ax, **kwargs):
    """
    TODO
    """
    ax.grid( linewidth=2.5, color='#AAAAAA', linestyle='--' )
    #maybe set x and y labels for all the plots with kwargs
    plt.rcParams.update( {'font.size':16} ) #might not work, to be tested
    return ax

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
    key.get_frame().set_linewidth(5 ) 
    if 'linewidth' in kwargs or 'lw' in kwargs:
        try: 
            key.get_frame().set_linewidth( kwargs['linewidth']) 
        except:
            key.get_frame().set_linewidth( kwargs['lw']) 
        finally:
            print( 'wrong format of linewidth specified, returns to the default')

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
