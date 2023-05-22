import numpy as np


class UniStuttgart:
    """
    Define collors as class attributes that they do not have to be called
    but can be accessed as attribute.
    This is intended to be used as
    from palette import UniStuttgart as uniS
    ... color=uniS.blue, ... etc.  
    """
    black       = '#000000'
    blue        = '#004191'
    lightblue   = '#00beff'
    lblue       = '#00beff'
    magenta     = '#ec008d'
    darkgreen   = '#00c832'
    green       = '#8dc63f'
    red         = '#ee1c25'
    yellow      = '#ffdd00'
    day9yellow  = '#ffa71a'
    orange      = '#f36f23'
    grey        = '#323232'
    grey10      = '#ececed'
    grey20      = '#d8dadb'
    grey30      = '#c5c7c9'
    grey40      = '#b2b4b7'
    grey50      = '#9fa1a5'
    grey60      = '#8b8f94'
    grey70      = '#787c82'
    grey80      = '#656970'
    grey90      = '#51575e'
    gray        = '#323232'
    gray10      = '#ececed'
    gray20      = '#d8dadb'
    gray30      = '#c5c7c9'
    gray40      = '#b2b4b7'
    gray50      = '#9fa1a5'
    gray60      = '#8b8f94'
    gray70      = '#787c82'
    gray80      = '#656970'
    gray90      = '#51575e'
    blue80      = '#3367a7'
    blue60      = '#668dbd'
    blue40      = '#99b3d3'
    lightblue80 = '#33cbff'
    lightblue60 = '#66d8ff'
    lightblue40 = '#99e5ff'
    lblue80     = '#33cbff'
    lblue60     = '#66d8ff'
    lblue40     = '#99e5ff'
    
    colorwheel = [lblue, day9yellow, green, blue, magenta, gray30, red ]
    color_wheel = colorwheel

    def __init__( self):
        """
        all colors are defined as class attributes not object instance attributes
        These colors can be directly called from the object as well as an object instance
        The following colors are defined:
    	- blue, blue80,60,40 
    	- lblue OR lightblue, lblue80,60,40 (analogue for shades)
    	- magenta
    	- green
    	- red
    	- yellow
    	- gray OR grey, gray10,20,30,40,50,60,70,80,90 (analogue for shades)
        """
        pass


def default_colorwheel():
    """
    Get the colors of the default colorwheel using this template
    """
    return uniS.colorwheel

def color_mixer( color_1, color_2, mixing_ratio=0.5, return_type='rgb'):
    """
    Take two RGB colors and mix them together by adding their colors together,
    e.g. color_1red*0.7 + color_2red * 0.3 = new_red ( for R G and B ) 
    The color can be given as a string in hex format (#abcdef)
    as a list/tuple in matplotlib with float values [ 0.1, 0.2, 1.0, alpha] 
    (where the opacity alpha will be ignored)
    or as a list/tuple with values ranging from 0-255 (every list/tuple 
    having values >1 will be assumed to be given in this format) 
    Parameters:
    -----------
    color_1:        tuple/list or string
                    color value of the first (weighted) color
    color_2:        tuple/list or string
                    color value of the second color 
    mixing_ratio:   float, default 0.5
                    weight 'color_1' has when mixing 
    return_type:    string, default tuple
                    specified return format, possible options
                    'tuple', 'list' (in float values 0-1 )
                    'hex', 'RGB' (with both reffering to hex RGB)
    Returns:
    --------
    mixed_color:    tuple/list or string
                    mixed color in the specified format 
    """
    ## first check in which format the colors have been given
    converted_colors = []
    for color in [color_1, color_2]:
        if isinstance( color, tuple) or isinstance( color, list):
            if (np.array( color) >1 ).any():
                color = [ x/255 for x in color]
            if len( color) == 4:
                color = list( color).pop() #remove the "alpha" channel of the color
        else: #convert hex string to list
            color           = color.replace( '#', '') #remove the "#"
            color_formatted = []
            for channel in range(3):
                color_value = int( color[channel*2: (channel+1)*2], 16)
                color_formatted.append( color_value/255.)
            color = color_formatted 
        converted_colors.append( np.array( color) )

    ## mix the colors and convert them to the desired return type
    mixed_color = converted_colors[0] * mixing_ratio + converted_colors[1] * (1-mixing_ratio)
    if return_type.lower() == 'tuple':
        return tuple( mixed_color)
    if return_type.lower() == 'list':
        return list( mixed_color)
    elif return_type.lower() == 'hex' or return_type.upper() == 'RGB':
        RGB =  '#'
        for channel in mixed_color:
            channel = int( channel * 255 )
            if channel <= 15:
                RGB += '0' + hex( channel)[-1:]
            else:
                RGB += hex( channel)[-2:]
        return RGB


################## SOON TO BE DEPECRATED ################
def CDColor( color_name='uniSblue'):
    """
    corporate design colors of uni stuttgart
    Parameters, optional:
    ---------------------
    color_name:     string, default uniSblue
                    chosen color 
    Returns:
    --------
    color_value:    string
                    RGB color value in hex format 
    available colors:                   
    - uniSblue                        
    - uniSblue80
    - uniSblue60
    - uniSblue40

    - uniSlblue OR uniSlightblue (analogue for shades)
    - uniSlblue80
    - uniSlblue60
    - uniSlblue40

    - uniSmagenta
    - uniSgreen
    - uniSred
    - uniSyellow
    - uniSgray OR uniSgrey (analogue for shades)
    - uniSgray10
    - uniSgray20
    - uniSgray30
    - uniSgray40
    - uniSgray50
    - uniSgray60
    - uniSgray70
    - uniSgray80
    - uniSgray90
    """ 
    color_name = color_name.lower()
    ### left case insensitive match
    if( color_name == 'unisblue' ):
        return '#004191'
    if( color_name == 'unislightblue'  or color_name == 'unislblue' ):
        return '#00beff'
    if( color_name == 'unismagenta' ):
        return '#ec008d'
    if( color_name == 'unisgreen' ):
        return '#8dc63f'
    if( color_name == 'unisred' ):
        return '#ee1c25'
    if( color_name == 'unisyellow' ):
        return '#ffdd00'
    if( color_name == 'unisgray' ):
        return '#323232'
    if( color_name == 'unisgray10' or color_name == 'unisgrey10' ):
        return '#51575e'
    if( color_name == 'unisgray20' or color_name == 'unisgrey20' ):
        return '#656970'
    if( color_name == 'unisgray30' or color_name == 'unisgrey30' ):
        return '#787c82'
    if( color_name == 'unisgray40' or color_name == 'unisgrey40' ):
        return '#8b8f94'
    if( color_name == 'unisgray50' or color_name == 'unisgrey50' ):
        return '#9fa1a5'
    if( color_name == 'unisgray60' or color_name == 'unisgrey60' ):
        return '#b2b4b7'
    if( color_name == 'unisgray70' or color_name == 'unisgrey70' ):
        return '#c5c7c9'
    if( color_name == 'unisgray80' or color_name == 'unisgrey80' ):
        return '#d8dadb'
    if( color_name == 'unisgray90' or color_name == 'unisgrey90' ):
        return '#ececed'
    if( color_name == 'unisblue80' ):
        return '#3367a7'
    if( color_name == 'unisblue60' ):
        return '#668dbd'
    if( color_name == 'unisblue40' ):
        return '#99b3d3'
    if( color_name == 'unislblue80' or color_name == 'unislightblue80' ):
        return '#33cbff'
    if( color_name == 'unislblue60' or color_name == 'unislightblue60' ):
        return '#66d8ff'
    if( color_name == 'unislblue40' or color_name == 'unislightblue40' ):
        return '#99e5ff'

