import matplotlib.pyplot as plt
import numpy as np
import plt_templates as template

## lines which can be modified
ls = template.linestyles()
dashed_lines = template.linestyles( ls='--', marker='x', mfc='yellow')
## adjustable parameters (which will affect figure size) (TRY IT OUT)
fontsize = 11.7 # default fontsize
ticksize = 9    # default ticksize
x_stretch = 6/6 # plot box is exactly 6cm wide, can be adjusted to be 7cm by e.g. x_stretch=7/6
y_stretch = 5/5 # analogue to x-stretch


plt.rcParams.update( template.rc_default( fontsize, ticksize) )


## some arbitrary data
x = [np.arange(1, 20)] *3
y = [2-np.log10(x[0])]
y.append( x[1]**2/ 20**2+1)
y.append( np.sqrt( x[2] )/4+0.2 )


## example on a single subplot using the predefined linestyles
fig, ax = template.fixed_plot( x_stretch=x_stretch, y_stretch=y_stretch)
ax.plot( x[0], y[0], label='log', **ls[0] )
ax.plot( x[1], y[1], label='square', **ls[2] )
ax.plot( x[2], y[2], label='square root', **dashed_lines[2] )
ax = template.add_legend( ax ) #has many default parameters built in
ax = template.axis_labels( ax, 'x [-]', 'y [-]' )
ax.set_title( 'Some nonlinear functions')
fig.savefig( 'functions.pdf')

## example on multiple subplots with legend position and stretch (size specification)
x_stretch = 7/6 
y_stretch = 4/5 
label_location= [ 'bot left', 'top left', 'upper right', 'lower right' ]
i = 0
fig, axes = template.fixed_plot( 2,2, x_stretch, y_stretch )
for ax in axes.flatten():
    ax.plot( x[0], y[0], label='log', **ls[0] )
    ax.plot( x[1], y[1], label='square', **ls[2] )
    ax.plot( x[2], y[2], label='square root', **dashed_lines[2] )
    ax = template.add_legend( ax, label_location[i] ) #has many default parameters built in
    ax = template.axis_labels( ax, 'x [-]', 'y [-]' )
    ax.set_title( 'Some nonlinear functions')
    i += 1
fig.savefig( 'multiplots.pdf')


## example on the default color palette ( CDColors() )
n_lines = 10
x = [0,0.3, 0.6, 1]
y = np.vstack( len(x)*[np.arange( 1, -1- 2/n_lines, -2/n_lines ) ] )
fig, ax = template.fixed_plot()
for i in range(n_lines):
    ax.plot( x, y[:,i], marker='o', label='{}th line'.format( i) )
ax = template.add_legend( ax, 'bot right') 
ax.set_xlim( xmax=1.9)
ax.set_title('default colors')
fig.savefig( 'default_colors.pdf' )
