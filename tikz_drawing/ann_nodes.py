import os
import sys
import numpy as np
from math import ceil, floor
from palette import UniStuttgart as uniS

### These functions print tikz strings to file and uses mostly bash related
### functions for this purpose. Might as well be a bash function but uses 
### python flexibility/arithmetic for convenience

def image_stack( to_file, images, pos=[0,0], increments=[0.5,0.5], slant=0, image_size=[2.5,2.5]):
    """ draw multiple stack like looking images 
    draws it such that the foreground image is at position <pos>"""
    n_images = len(images )
    pos_max = n_images * np.array( increments)//2  + np.array( pos) #center such that x is on 1/2
    image_node    = "  \\\\node (stack_img{})at ({}, {}) [yslant={}] {{\\\\includegraphics[width={}cm, height={}cm]{{{}}}}};'"
    os.system( "echo '\\\\begin{{scope}}[yslant={}]'".format(slant) + to_file )
    for i in range( n_images):
        current_pos = pos_max - i*np.array(increments) #(n_images - (i+1) )*np.array(increments)
        os.system( "echo '"+ image_node.format( i, *current_pos, slant, *image_size, images[i]) + to_file )
    os.system( "echo '\\\\end{scope}'" + to_file ) 


def dense_model( to_file, start_coos=None, neurons=None, inf_hidden=3, n_inf=None, n_dots=3, **kwargs):
    """
    Echo a generic dense model from x_start using <len(neurons)> hidden
    layers and display the amount of specified neurons
    The model here/size ratios are deemed good enough so when a small 
    model is needed it should be \scaled in a scope
    Parameters:
    -----------
    to_file:    str
                pipe into file, should look similar to  '>> ann.tex'
    start_coos: list like of floats, default None
                (x_start,y_middle) coordinates, defaults to ( neuron_width, middle_y_req)
    neurons:    list of ints and 'inf', default None
                how many neurons per layer, layers given as
                [input, *n*[hidden], output layer]
                defaults to ['inf', *3*['inf'], 3]
    inf_hidden: int, default 3
                if the middle hidden layer should be turned into 'inf'
                dots (and whited out). False if not
    n_inf:      int, default None
                how many hidden neurons on each side of the 'inf' dots
                defaults to [3, *3*[4], 2]
    n_dots:     int, default 3
                how many black dots to indicate infinite length
    **kwargs:   kwargs with default arguments 
    lines:          str, default 'uniSblue' #latex compatible colors
    neuron_in:      str, default 'uniSgreen'
    neuron_out:     str, default 'uniSgreen'
    neuron_hidden:  str, default 'uniSgreen'
    dots:           black #TODO put default colors as argument
    node_label:     formattable string, default 'n{}{}'
    layer_distance: float, default 2.5
    rel_distance:   float, default 1.2 #relative size of space w.r.t. node size
    node_size:      float, default 0.75
    line_width:     float, default 1.0
    rotation:       float, default 0 #required if the ann is rotated and inf_hidden drawn
    extra_neuron:   str, default None, color of an extra input neuron
    Returns:
    --------
    None:       only prints the tikz things <to_file>
    """
    ## absolute sizes
    ## relative sizes
    neuron_scale   = 0.6 #have to use scale to get neurons smaller
    dot_size       = 0.5 #relative size to node_size
    node_distance  = 1.2 #factor of size of the neurons
    scope_end      = "echo '\\\\end{scope}'" + to_file
    ## input processing - recomputation of variables
    #other inputs
    neurons              = neurons if neurons is not None else  ['inf', *3*['inf'], 3]
    n_inf                = n_inf if n_inf is not None else [3, *3*[4], 2]
    n_inf                = n_inf + max( 0, len(neurons) - len(n_inf) )*[None]
    neurons              = neurons.copy()
    height_determination = neurons.copy()
    for i in range( len(height_determination)):
        if height_determination[i] is 'inf':
            height_determination.pop(i)
            print( n_inf[i], n_dots )
            height_determination.insert( i, 2*n_inf[i] + n_dots)
    max_neurons = max( height_determination)
    #kwargs, color and node label is forwarded to 'dense_layer'
    layer_distance  = kwargs.pop( 'layer_distance', 2.5)
    rel_distance   = kwargs.pop( 'rel_distance', 1.2)
    node_size       = kwargs.pop( 'node_size', 0.75 )
    line_width      = kwargs.pop( 'line_width', 1.0)
    rotation        = kwargs.pop( 'rotation', 0)
    extra_neuron    = kwargs.pop( 'extra_neuron', None)
    color = [kwargs.pop( 'neuron_in', 'uniSlightblue' )]
    color.extend( (len(neurons)-2) * [kwargs.pop( 'neuron_hidden', 'uniSgreen' )] )
    color.append( kwargs.pop( 'neuron_out', 'uniSblue' ) )
    line_color = kwargs.pop( 'lines', 'uniSblue')
    node_label = kwargs['node_label'] if 'node_label' in kwargs else 'n{}{}' 
    #computations
    connection_width = line_width/2
    node_distance    = node_size * neuron_scale * rel_distance
    dot_size        *= neuron_scale
    y_total          = max_neurons * node_distance + node_distance
    y_middle         = start_coos[1] if start_coos else y_total/2
    x_current        = start_coos[0] if start_coos else node_distance
    # infinite white rectangle in the middle
    inf_idx =  len(neurons)//2  if inf_hidden else -1
    inf_pos = x_current + inf_idx*layer_distance 

    ## echo strings
    neuron_scope = "\\\\begin{{scope}}[circle, fill, draw=uniSblue, minimum size={}cm, line width={}pt]'".format( node_size, line_width)
    neuron       = "  \\\\node ({{}}) at ({{}}, {{}}) [draw, fill={{}}, scale={}]{{{{}}}};'".format( neuron_scale)
    dot          = "  \\\\node at ({{}}, {{}}) [circle, fill=uniSgray, scale={}]{{{{}}}};'".format( dot_size) 

    ## all the layers put in the single list
    os.system( "echo '" + neuron_scope + to_file)
    for i in range( len(neurons)): 
        if i == 0 and extra_neuron:
            dense_layer( neuron, dot, neurons[i], x_current, y_middle, i, color[i],
                    to_file, node_distance, n_inf[i], n_dots, extra_neuron=extra_neuron, **kwargs)
        else:
            dense_layer( neuron, dot, neurons[i], x_current, y_middle, i, color[i],
                    to_file, node_distance, n_inf[i], n_dots, **kwargs)
        neurons[i]  = neurons[i] if neurons[i] is not 'inf' else 2*n_inf[i]
        x_current  += layer_distance
    os.system( scope_end) 

    #connections of the dense model
    os.system( "echo '\\\\begin{{scope}}[draw=uniSblue, line width={}pt]'".format(connection_width) +to_file )
    for i in range( len(neurons) -1 ):
      for j in range( neurons[i] ):
        for k in range( neurons[i+1]):
          if i==0 and extra_neuron:
              from_node = node_label.format( i, neurons[0])
              to_node   = node_label.format( i+1, k)
              os.system( "echo '  \\\\draw ({}) to ({});' ".format( from_node, to_node) + to_file ) 
          from_node = node_label.format( i,j)
          to_node   = node_label.format( i+1, k)
          os.system( "echo '  \\\\draw ({}) to ({});' ".format( from_node, to_node) + to_file ) 
    os.system( scope_end)
    ## white box in the middle + inf dots
    y_total -= node_distance #correction because the nodes are not 'drawn'
    y_middle += node_distance/2
    if inf_hidden and len(neurons) >= 5:
        os.system( "echo '" + neuron_scope + to_file)
        width     = max(1.0*layer_distance, (1+inf_hidden)*node_distance )
        rectangle = '\\\\node at ({},{})  [rotate={}, rectangle, fill=white, minimum width={}cm, minimum height={}cm]{{}};'.format( inf_pos, y_middle, rotation, width, y_total)
        os.system( "echo '"+rectangle + "'" + to_file )
        x_current = inf_pos - (inf_hidden/2-0.5)*node_distance
        for i in range( inf_hidden):
            for y in [ y_middle - y_total/3, y_middle, y_middle + y_total/3]:
                os.system( "echo '" + dot.format( x_current, y ) + to_file )
            x_current  += node_distance 
        os.system( scope_end)
    return


def dense_layer( node, dot, neurons, x_pos, y_middle, layer_nr, color, to_file, node_distance, n_inf=4, n_dots=3, extra_neuron=None, **kwargs ):
    """
    echo one layer of console to file, parameters are similar to those above 
    Additionally needs the 'node' and 'dot' which are formattable
    Kwargs are the same as in the function above minus the line color
    Parameters:
    -----------
    node:           formattable string
                    string to format for the node
    dot:            formattable string
                    string to format for the inf cots
    neurons:        int or 'inf'
                    how many neurons to display in the layer
    x_pos:          float
                    x position of the current layer
    y_middle:       float
                    centered y_position of the current layer
                    NOTE: there is a slight bug/offset implemented
    layer_nr:       preferably int
                    how to format the layer label
    color:          str
                    tikz/tex compatible color
    to_file:        str
                    piping command to the output file, e.g. >> output_file.tex
    node_distance:  float
                    distance between two neurons
    n_inf:          int, default 4
                    how many neurons at each side of the inf dots
    n_dots:         int, default 3
                    how many 'inf' dots, only needed if <neurons> is 'inf' 
    extra_neuron:   str, default None
                    if desired puts an extra neuron at the bottom in a different color
                    does not offset the neurons above 
    **kwargs:
    takes the one from the 'dense_model', but only 'node_label' is used thus far
    """
    ## input procsesing
    label = kwargs['node_label'] if 'node_label' in kwargs else 'n{}{}' 
    ## drawing of connections
    if neurons is not 'inf': #simply draw all neurons in a layer
        y_current = y_middle + neurons * node_distance /2
        for j in range( neurons):
            node_label = label.format( layer_nr,j) 
            os.system( "echo '" + node.format( node_label, x_pos, y_current, color ) + to_file )
            y_current  -= node_distance
    else: #draw n_inf neurons at the top and bottom and draw n_dots in the middle
        y_current = y_middle + (n_inf*2+n_dots) * node_distance /2
        for j in range( n_inf): 
            node_label = label.format( layer_nr,j) 
            os.system( "echo '" + node.format( node_label, x_pos, y_current, color ) + to_file )
            y_lower    = y_current-(n_inf+n_dots)*node_distance
            node_label = label.format( layer_nr,j+n_inf) 
            os.system( "echo '" + node.format( node_label, x_pos, y_lower, color  ) + to_file )
            y_current  -= node_distance
        for j in range( n_dots):
            os.system( "echo '" + dot.format( x_pos, y_current ) + to_file )
            y_current  -= node_distance
    if extra_neuron:
        j = 2*n_inf if neurons is 'inf' else neurons
        y_current = y_lower - node_distance if  neurons is 'inf' else y_current
        node_label = label.format( layer_nr, j )
        os.system( "echo '" + node.format( node_label, x_pos, y_current, extra_neuron ) + to_file ) 
    return


def connect_to( to_file, from_label, to_label, iteration_nr=None, scope_options=None, draw_options='color=uniSblue, line width=0.5pt'):
    """ 
    connect 'from_label' 'to_label', if 'to_label' is a formattable
    string then 'iteration_nr' has to be an int, and specifies the 
    range to iterate over. If scope options are passed a scope is invoked,
    otherwise it draws with default thin connections. Only _really_ makes 
    in combination with iteration_nr. 
    Parameters:
    -----------
    to_file:        str
                    string which pipes to file
    from_label:     str
                    node to start the connection from
    to_label:       str or formattable string
                    where to connect, if formattable 'iteration_nr' has to be given
    iteration_nr:   int, default None
                    up to which int to loop to the 'to_label'
    scope_options:  str, default None
                    options of tikz scope, if given a scope is invoked
    draw_options:   str, default 'color=uniSblue, line width=0.5pt'
                    options for drawing, is only considered if scope_options is None 
    """
    ## input catching, if a scope is desired
    if scope_options is None:
        connection = "\\\\draw [{}] ({{}}) to ({{}});'".format( draw_options)
    else:
        os.system( "echo '\\\\begin{scope}[{}]'".format(scope_options) + to_file )
        connection = "\\\\draw  ({}) to ({});'"
    ## draw multiple or single lines
    if iteration_nr is None:
      os.system( "echo '  " + connection.format( from_label, to_label) + to_file )
    else:
      for i in range( iteration_nr):
        os.system( "echo '  " + connection.format( from_label, to_label.format(i)) + to_file )
    ## terminate scope if specified
    if scope_options is not None:
        os.system( "echo '\\\\end{scope}'" + to_file )
    return

##################### Convolutional neural network #########################

def conv_net( to_file, start_coos=None, channels=None, scale=None, input_image=None, connection_labels=None, **kwargs):
    """
    if <input_image> is given then 'connection_labels' has to be 1 longer, to consider the connection from the input layer.
    Parameters:
    -----------
    to_file:        str
                    pipe into file, should look similar to  '>> ann.tex'
    start_coos:     list like of floats, default None
                    tikz position of the conv net, defaults to left most ~middle
    channels:       list like of ints, default None
                    how many channels per layer, defaults to [
    scale:          list like of floats, default None
                    scale of the current layer, simulates downscaling
                    defaults to
    input_image:    str, default None
                    path to full file, if given an input image will be displayed
    connection_labels:  list of str, default None
                        possible labels to put above the drawn connections between layers
                        if True is given it defaults to something sensible for the default 
                        parameters above
    **kwargs with default arguments
    layer_label:        formattable string, default c{}{}{{}}
                        label for each layer, CARE it has to be twice formattable
    colors:             color scheme, TO BE IMPLEMENTED
    channel_size:       float, default 2.5 
    channel_distance:   float, default 0.2 
    layer_distance:     float, default 1.0
    slant:              float, default 0.8
    Returns:
    --------
    None: only prints <to_file>
    """
    ## input processing, taking default kwargs
    channel_size  = kwargs.pop( 'channel_size', 2.5)
    channel_distance  = kwargs.pop( 'channel_distance', 0.20)
    layer_distance   = kwargs.pop( 'layer_distance', 1.0)
    slant        = kwargs.pop( 'slant', 0.8)
    layer_label = kwargs.pop( 'layer_label', 'c{}{}{{}}' )
    scale = scale if scale is not None else [1, 0.5, 0.25, 1/8, 1/8]
    channels = channels if channels is not None else [4, 12, 12, 32, 16]
    if kwargs:
        raise Exception( 'uncaught kwargs passed to tikz_generators.dense_model, terminating code for typo catching, keys are {}'.format( kwargs.keys() ) )
    ## no labels or default labels
    if connection_labels is None:
        connection_labels = (len(channels)+1)*[None]
    elif connection_labels is True:
        connection_labels = [ 'conv 5/1$\\\\times${}'.format( channels[0]),'conv 3/2$\\\\times$12'.format( channels[1]),
            'pool 2/2'.format( channels[2]),'conv 3/2$\\\\times$32'.format( channels[3]),
            'conv 1/1$\\\\times$16'.format( channels[4]) ]
        if input_image:
            connection_labels.pop(0)
    ### echo string allocations
    scope_end     = "echo '\\\\end{scope}'" + to_file
    connection    = '  \\\\draw ({}{}e) to ({}{}s);'
    image_node    = "  \\\\node at ({}, {}) [yslant={}] {{\\\\includegraphics[width={}cm, height={}cm]{{{}}}}};'"
    ## computation of inputs
    width = round( channel_size/(1+slant), 6)
    height = channel_size
    y_total = height + 2*slant
    y_middle   = start_coos[1] if start_coos else y_total/2
    x_current  = start_coos[0] if start_coos else slant
    size = np.array( [width, height]) 
    layer_kwargs = dict( to_file=to_file, base_size=size, slant=slant, change_pos=True)

    ## draw the input image
    if input_image:
        os.system( "echo '"+image_node.format( x_current, y_middle, slant, width, height, input_image) + to_file )
        conv_coordinates( size, [x_current, y_middle], slant, layer_label.format( 0, 'end'), to_file )
        x_current += layer_distance
    current_size = scale[0]* size
    current_pos = [x_current, y_middle]
    if input_image:
        conv_coordinates( current_size, current_pos, slant, layer_label.format( 1, 'start'), to_file )
        conv_connections( layer_label.format( 0, 'end'), layer_label.format( 1, 'start'), to_file, connection_labels.pop(0) )
    ## draw all convolutional layers with connections
    current_size = scale[0]* size
    current_pos = [x_current, y_middle]
    for i in range( len( channels) -1):
        conv_layer( channels[i], current_pos, scale=scale[i], **layer_kwargs)
        conv_coordinates( size, current_pos, slant, layer_label.format( i+1, 'end'), to_file, scale=scale[i] )
        # for next channel, also draw connections for visual order
        current_pos[0] += scale[i+1]**(1/2.0) *layer_distance 
        conv_coordinates( size, current_pos, slant, layer_label.format( i+2, 'start'), to_file, scale=scale[i+1] )
        conv_connections( layer_label.format( i+1, 'end'), layer_label.format( i+2, 'start'), to_file, connection_labels[i] )
        #conv_layer( channels[i], current_pos, to_file, scale=scale[i], base_size=current_size, slant=slant)
        #x_current += (channels[i]-1) * scale[i]*channel_distance
        #conv_coordinates( current_size, [x_current, y_middle], slant, layer_label.format( i+1, 'end'), to_file )
        ## for next channel, also draw connections for visual order
        #x_current += scale[i+1]**(1/2.0) *layer_distance 
        #current_pos = [x_current, y_middle]
        #current_size = scale[i+1] * size
        #conv_coordinates( current_size, current_pos, slant, layer_label.format( i+2, 'start'), to_file )
        #conv_connections( layer_label.format( i+1, 'end'), layer_label.format( i+2, 'start'), to_file, connection_labels[i] )
    i += 1
    conv_layer( channels[i], current_pos, scale=scale[i], **layer_kwargs)
    #x_current += (channels[i]) * channel_distance**scale[i]**0.6
    conv_coordinates( size, current_pos, slant, layer_label.format( i+1, 'end'), to_file, scale=scale[i] )
    #middle_coos = layer_label.format( i+1, 'end').format( 'M') #middle of last conv channel
    return


def conv_layer( n_channel, pos, to_file, scale=1, base_size=2.5, base_distance=0.2, slant=0.8, echo_coos=False, change_pos=False  ):
    """ 
    draw a convolutional layer of mutiple channels with current scale
    Parameters:
    -----------
    pos:        list like of floats
                center x and y position of the first channel
    to_file:    str
                some string which pipes to file, e.g. '>> tex_file.tex'
    scale:      float, default 1
                is technically inverse the downscaling factor, should be e.g. 1/2 (for a 2strided conv)
                Does have and effect on the 'base' variables
    base_size:  list like of floats, or float, default 2.5
                either width & height or only height and width is computed (assuming squares)
    base_distance: float, default 0.2
                distance between the channels
    slant:      float, default 0.8
                how much the conv channels should be tilted/slanted 
    echo_coos:  int, default False
                Whether to echo the coordinates of the start and end channel
                The coordinates will be labeled 'c{n_layer}{start/end}{corner/Middle} 
    change_pos: bool, default False
                whether the passed position should be changed in place
    Returns:
    --------
    None        only prints <to_file>
    """
    if isinstance( base_size, float):
        width = round( base_size/(1+slant), 6)
        base_size = [ width, base_size]
    size = [ x*scale for x in base_size]
    channel_distance = base_distance*scale**0.6
    channel_scopes = "\\\\begin{scope}[rectangle, fill=uniSblue!60!uniSlightblue, draw=uniSblue, opacity=0.5, "  
    current_size  = "minimum width={}cm,  minimum height={}cm]'"
    node          = "  \\\\node at ({}, {}) [yslant={}, draw, fill] {{}};'"
    os.system( "echo '" + channel_scopes + current_size.format( *size) + to_file  )
    ## echo all filters
    if change_pos is False:
        pos = list(pos).copy()
    if echo_coos is not False:
        node_label = f'c{echo_coos}{"start"}{{}}'
        conv_coordinates( size, pos, slant, node_label, to_file)
    
    for j  in range( n_channel):
        os.system( "echo '" + node.format( *pos, slant) +to_file)
        if j != n_channel-1:
            pos[0] += channel_distance
    if echo_coos is not False:
        node_label = f'c{echo_coos}{"end"}{{}}'
        conv_coordinates( size, pos, slant, node_label, to_file)
    os.system( "echo '\\\\end{scope}'" + to_file ) 


def conv_coordinates( size, pos, slant, node_label, to_file, scale=1 ):
    """
    Give all edge coordinates of the channel of current <size> at current
    <pos>ition. The edges are enumerated as TR: A, BR: B, TL: C, BL: D, middle: M, mid bot E
    Parameters:
    -----------
    size:       list like of floats
    pos:        list like of floats
    slant:      float
                yslant value in tikz
    node_label: formatable string
                e.g. c0{}end
    to_file:    str, where to pipe the console output
    """
    size = [ x*scale for x in size]
    pos_x = 2*[pos[0] + size[0]/2]  + 2*[pos[0] - size[0]/2] + [pos[0]] + [pos[0]]

    y_0   = pos[1] - pos[0] * slant
    pos_y = [y_0 + size[1]/2]  + [y_0 - size[1]/2] +[y_0 + size[1]/2]  + [y_0 - size[1]/2] + [y_0] + [y_0 - size[1]/2]
    os.system( "echo '\\\\begin{{scope}}[yslant={}]'".format(slant) + to_file )
    corners = 'ABCDME' #corners and middle
    for i in range( len(corners)):
        label = node_label.format( corners[i])
        os.system( "echo '  \\\\coordinate ({}) ".format( label) + 'at ({}, {}) '.format(pos_x[i], pos_y[i] ) + "{};'" + to_file )
    os.system( "echo '\\\\end{scope}'" + to_file )


def conv_connections( from_node, to_node, to_file, label=None, corners='ABCD', scope_kwargs=''):
    """ draw the connections of all for edges between the conv layers 
    Adds a label to the top right corner connection if given
    For good results this should be called before the layer is drawn
    such that the layer overlaps the connection, (i don't think tikz has
    zorder)
    Should be called in combination with the 'conv_coordinates' function
    Parameters:
    -----------
    from_node:  formattable str
                node label from the previous convolution layer 
    to_node:    formattable str
                node label from the previous convolution layer 
    to_file:    string 
                string to pipe in the echo function
    label:      string, default None
                appanrelty a label to put on the top left/right corner
                haven't used that one yet, and i don't know for sure
    corners:    string, default ABCD
                which corners to connect, defaults to outers,
                'M' for only middle connection 
    scope_kwargs:   string, default ''
                    what to add in the kwargs of the scope for all connections
    """
    if scope_kwargs and scope_kwargs[0] not in [',', ' ']:
        scope_kwargs = ', ' + scope_kwargs
    if corners == 'M':
        os.system( f"echo '\\\\begin{{scope}}[draw=uniSblue, triangle 90 cap reversed-triangle 90 cap, line width=2.5pt, font=\\\\tiny{scope_kwargs}]'"  + to_file) 
    else:
        os.system( f"echo '\\\\begin{{scope}}[draw=uniSblue, line width=0.68pt, font=\\\\tiny{scope_kwargs}]'"  + to_file) 
    connection    = "  \\\\draw ({}) to ({});'"
    text_connection = "  \\\\draw ({}) to node[midway, above, sloped] {{{}}} ({});'"
    for corner in corners:
        start = from_node.format( corner)
        end = to_node.format( corner)
        if label is not None and corner == 'A':
          os.system( "echo '" + text_connection.format( start, label, end) + to_file)
        else:
          os.system( "echo '" + connection.format( start, end) + to_file)
    os.system( "echo '\\\\end{scope}'" + to_file ) 

###### dependency functions/layer generators ####
