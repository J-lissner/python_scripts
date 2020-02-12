import numpy as np

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

    if( color_name == 'uniSblue' ):
        return '#004191'
    if( color_name == 'uniSlightblue'  or color_name == 'uniSlblue' ):
        return '#00beff'
    if( color_name == 'uniSmagenta' ):
        return '#ec008d'
    if( color_name == 'uniSgreen' ):
        return '#8dc63f'
    if( color_name == 'uniSred' ):
        return '#ee1c25'
    if( color_name == 'uniSyellow' ):
        return '#ffdd00'
    if( color_name == 'uniSgray' ):
        return '#323232'
    if( color_name == 'uniSgray10' or color_name == 'uniSgrey10' ):
        return '#51575e'
    if( color_name == 'uniSgray20' or color_name == 'uniSgrey20' ):
        return '#656970'
    if( color_name == 'uniSgray30' or color_name == 'uniSgrey30' ):
        return '#787c82'
    if( color_name == 'uniSgray40' or color_name == 'uniSgrey40' ):
        return '#8b8f94'
    if( color_name == 'uniSgray50' or color_name == 'uniSgrey50' ):
        return '#9fa1a5'
    if( color_name == 'uniSgray60' or color_name == 'uniSgrey60' ):
        return '#b2b4b7'
    if( color_name == 'uniSgray70' or color_name == 'uniSgrey70' ):
        return '#c5c7c9'
    if( color_name == 'uniSgray80' or color_name == 'uniSgrey80' ):
        return '#d8dadb'
    if( color_name == 'uniSgray90' or color_name == 'uniSgrey90' ):
        return '#ececed' 
    if( color_name == 'uniSblue80' ):
        return '#3367a7'
    if( color_name == 'uniSblue60' ):
        return '#668dbd'
    if( color_name == 'uniSblue40' ):
        return '#99b3d3'
    if( color_name == 'uniSlblue80' or color_name == 'uniSlightblue80' ):
        return '#33cbff'
    if( color_name == 'uniSlblue60' or color_name == 'uniSlightblue60' ):
        return '#66d8ff'
    if( color_name == 'uniSlblue40' or color_name == 'uniSlightblue40' ):
        return '#99e5ff'



def color_mixer( color_1, color_2, mixing_ratio=0.5, return_type='rgb'):
    """
    Take two RGB colors and mix them together by adding their colors together,
    e.g. color_1red*0.7 + color_2red * 0.3 = new_red ( for R G and B )

    The color can be given as a string in hex format (#abcdef)
    as a list/tuple in matplotlib with float values [ 0.1, 0.2, 1.0, alpha] (where the opacity alpha will be ignored)
    or as a list/tuple with values ranging from 0-255 (every list/tuple having values >1 will be assumed to be given in this format)

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
            color = color.replace( '#', '') #remove the "#"
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



