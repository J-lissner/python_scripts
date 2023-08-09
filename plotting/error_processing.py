import numpy as np 
import matplotlib.pyplot as plt
from math import ceil, floor
from matplotlib.colors import LinearSegmentedColormap as Cmap
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

import plt_templates as plot
import general_functions as func
from general_functions import Cycler
from palette import UniStuttgart as uniS
from palette import color_mixer

plt.rcParams.update( plot.rc_defaults() )

def r_squared_plot( ax, solution, prediction, error_bounds=[1,2,5,10], box_loc=None, rsquare_number=True, **axset ):
    """
    Put the R^2 plot into the given axis. Additionally puts in decorating
    lines which indicate the relative errors.
    Modifies the ax object in place, does not have to be overwritten
    Parameters:
    -----------
    ax:             matplotlib.pyplot ax object
                    axes with the added R^2 plot and decorators
    solution:       numpy 1d-array
                    true target values
    prediction:     numpy 1d-array
                    predicted values of the model
    error_bounds:   list of int/float, default [1,2,5,10]
                    decorating lines to indicate the relativ error 
    box_loc:        tuple of 4 floats, default None
                    if given a market inset will be drawn, the parameters of the box_loc
                    are 'bottom, left, width, height' of the box (positions relative
                    to figure, care!
                    If True is passed, it is assumed that its a single ax figure from
                    the plt_template and will locate it accordingly
    rsquare_number: bool, default True
                    if a text decorator should give the rsquare score inside the plot
    axset:          kwargs with default arguments, called in the ax.set( **axset) method
        xlabel:     'true value'
        ylabel:     'predicted value'
    Returns:
    --------
    ax:     matplotlib.pyplot ax object
            axes with the added R^2 plot and decorators
    """
    if box_loc is True:
        box_loc = (2/3+0.02, 0.18, 0.27, 0.25)
    ## input preprocessing
    axset['xlabel'] = axset.pop( 'xlabel', 'true value' )
    axset['ylabel'] = axset.pop( 'ylabel', 'predicted value' )
    ## Preprocessing and allocations
    xmin = min( prediction.min(), solution.min() )
    xmax = max( prediction.max(), solution.max() )
    if min( xmin, -xmax) < 0:
        error_anchor = [ 2*min( xmin, -xmax), 2*xmax]
    elif np.allclose( min( xmin, -xmax), 0, atol=5e-2):
        error_anchor = [ -0.5, 2*xmax]
    else:
        error_anchor = [ 0.5*min( xmin, -xmax), 2*xmax]
    box_color = (0.9,0.9,0.9,0.3) #some transparent gray
    error_colors = Cycler( [uniS.blue, uniS.lblue, uniS.green, uniS.orange, uniS.red, uniS.yellow ])
    lw= 1.5
    ## local computations
    rsquare = 1 - ((solution-prediction)**2).sum(0) /((solution-solution.mean(axis=0) )**2).sum(0)
    ## plotting of data
    ### color based on relative error
    color = np.zeros( prediction.shape).astype('object')
    n_errors = len( error_bounds)
    color[:] = error_colors[n_errors-1]
    base_kwargs = dict(linewidth=0.1, edgecolor='none')
    for current_color, error in reversed( list( zip( error_colors[:n_errors], np.sort( error_bounds) ))) :
        color_idx = ( np.abs(solution-prediction)/np.maximum(np.abs(solution), 1e-8) ) <= error/100
        if error != error_bounds[-1]:
            label = f'$\leq${error}\% error' 
        else:
            label = f'$>${error_bounds[-2]}\% error' 
        ax.scatter( *2*[-1e10], color=current_color, **base_kwargs, s=20, label=label)
        color[ color_idx] = current_color 
    ## zoom in on the middle part
    dx = xmax-xmin
    weight_min = lambda x: 0.875* x if x < 0 else 1.125*x
    weight_max = lambda x: 0.875* x if x > 0 else 1.125*x
    box_lims = [xmin + 0.425*dx, xmax-0.425*dx]
    #box_lims = [ weight_min(xmin), weight_max( xmax) ]

    ## plotting of data
    ax.scatter( solution, prediction, color=color, **base_kwargs, s=3, zorder=2) #, edgecolor='k', )
    ax.plot( error_anchor, error_anchor, color=uniS.black, lw=lw, zorder=1 ) 
    # box
    if box_loc is not None:
        box = plt.axes( box_loc )
        box.scatter( solution, prediction, color=color, **base_kwargs, s=3, zorder=2) #, edgecolor='k', )
        box.plot( error_anchor, error_anchor, color=uniS.black, lw=lw, zorder=1 )
        plt.setp( box, xticks=[], yticks=[] )
        box.set_xlim( box_lims)
        box.set_ylim( box_lims)
        mark_inset( ax, box, loc1=1, loc2=3, facecolor=box_color, edgecolor='k', lw=1.5)
    #decorations
    if rsquare_number is True:
        ax.text( 0.48, 0.92, r'$R^2\approx {:1.5f}$'.format( rsquare),  transform=ax.transAxes )
    if xmin < 0:
        ax.set_xlim( xmin=1.05*xmin, xmax=1.05*xmax)
        ax.set_ylim( ymin=1.05*xmin, ymax=1.05*xmax)
    else:
        ax.set_xlim( xmin=0.95*xmin, xmax=1.05*xmax)
        ax.set_ylim( ymin=0.95*xmin, ymax=1.05*xmax)
    ax.set( **axset)
    plot.add_legend( ax, position='top left')
    return ax



def binned_errors( x, y, y_pred, n_bins=15, line_threshold=None, error_measure='component wise', quantiles=[0.99, 0.95, 0.90], extra_legend='figures/bin_error_legend.pdf', ylabel=None):
    """
    Scatter the data and the data prediction and set them in relation
    with their error plotted on the -y-axis. The errors are shown for
    each bin below the 0-value of the y-axis, shows max,mean and median.
    Only works nicely for data with values > 1.
    Parameters:
    -----------
    x:          numpy 1d-array
                samples of the input parameter (built for volume fraction)
    y:          numpy 1d-array
                samples of the target value (built for heat conductivity)
    y_pred:     numpy 1d-array
                predicted target values
    n_bins:     int, default 15
                how many error bars should be shown at the bottom
    line_threshold: float, default None
                    draw lines between corresponding samples. Only samples with 
                    error > line_threshold will be connected by lines 
    error_measure:  str, default 'component wise'
                    what error measure to choose #TODO, right now its mse per component
    quantiles:      list like of ints, default [0.9, 0.8]
                    maximum error bars of quantiles. For the plot to look nice they
                    must be larger than 50 and sorted descendingly. supports at most 5 quantiles
    extra_legend:   str, default 'figures/bin_error_legend.pdf'
                    if an auxiliary legend should be saved outside of the plots
    ylabel:         list of 3 str, default None
                    list of the ylabels, defaults to the thermal problem
    """
    ## Error computation and error binning
    error = y-y_pred
    n_samples = x.shape[0]
    n_quantiles = len( quantiles) 
    ylabels = [r'$\bar\kappa_{11} [-]$', r'$\bar\kappa_{22} [-]$', r'$\sqrt2\bar\kappa_{12} [-]$' ] if ylabel is None else ylabel
    if error_measure == 'component wise':
        n_target = y.shape[1]
        error = np.sqrt( error**2 )
    ## get the indices of the corresponding values per bin
    sorted_idx, bins = func.bin_indices( x, n_bins=n_bins)
    bin_incr = bins[1]- bins[0]
    max_errors = np.zeros( ( n_target, n_bins) )
    mean_errors = np.zeros( ( n_target, n_bins) )
    median_errors = np.zeros( ( n_target, n_bins) )
    quantile_errors = np.zeros( (n_target, n_bins, n_quantiles) )
    for i in range( n_bins -1):
        bin_errors = error[sorted_idx[i]].squeeze()
        max_errors[:,i] = bin_errors.max( 0)
        mean_errors[:,i] = bin_errors.mean( 0)
        median_errors[:,i] = np.median( bin_errors, 0)
        for j in range( n_target):
          for k in range( n_quantiles):
            quantile_errors[j,i,k] = np.sort( bin_errors[:,j])[ floor( quantiles[k]* len(bin_errors[:,j])) ]  
    ## find the samples which have a large error to draw lines between
    if line_threshold is not None:
        error_lines = []
        for i in range( n_target):
            error_lines.append( np.argwhere( error[:,i] >= line_threshold).squeeze()) 
    ## find all samples to plot, plot all maximum error samples in addition to the  'incr' random samples
    incr = 1#5
    n_max = 10
    plotted_samples = list( range( 0, n_samples, incr) )
    absolute_errors = np.abs(error)
    for i in range( error.shape[-1] ): #component wise max absolute error
        plotted_samples += list( np.argsort( absolute_errors[:,i] )[-n_max:] ) 
    mse = (error**2 ).mean(1)
    plotted_samples += list( np.argsort( mse)[-n_max:])
    rel_mse = np.linalg.norm( error, axis=1) / np.linalg.norm( y,axis=1)
    plotted_samples += list( np.argsort( rel_mse)[-n_max:])
    plotted_samples = list( set( plotted_samples ) )



    fig, axes = plot.fixed_plot(1, n_target, x_stretch=1.3) 
    reference_style = dict(s=15, color=uniS.lblue, edgecolor='black')
    #mixred = color_mixer( uniS.day9yellow, uniS.red )
    prediction_style = dict(s=4, linewidth=0.002, color=uniS.day9yellow, edgecolor=uniS.red) #slightly smaller than true values
    quantile_colors = [uniS.gray, uniS.gray80, uniS.gray60, uniS.gray40, uniS.gray20 ]# uniS.gray60, uniS.gray70, uniS.gray80, uniS.gray90, uniS.gray100 ]
    quantile_label = ( len(quantiles) ) * '{}, ' + '{}-quantiles'
    quantile_label = quantile_label.format( *quantiles[::-1], 1) 
    for i in range( n_target):
        bar_kwargs = dict( width=bin_incr, bottom=min(0, 1.01*y[:,i].min() ))
        axes[i].scatter( x[plotted_samples], y[plotted_samples,i], **reference_style, label='target values')
        axes[i].scatter( x[plotted_samples], y_pred[plotted_samples,i], **prediction_style,  label='predicted values')
        if quantiles:
            axes[i].bar( bins + bin_incr/2, -max_errors[i], **bar_kwargs, color=quantile_colors[0] ) #(1 quantile)
        ## error quantiles 
        for j in range( len( quantiles) ):
            axes[i].bar( bins+bin_incr/2, -quantile_errors[i,:,j], **bar_kwargs, color=quantile_colors[j+1] )
        #axes[i].bar( bins + bin_incr/2, -mean_errors[i], **bar_kwargs, color=uniS.green, label='mean error' )
        #axes[i].bar( bins + bin_incr/2, -median_errors[i], **bar_kwargs, color=uniS.blue, label='median error' )
        if line_threshold is not None:
            for sample in error_lines[i]:
                x_pos = 2*[ x[sample] ]
                y_pos = [ y[sample,i], y_pred[sample,i] ]
                axes[i].plot( x_pos, y_pos, color=uniS.yellow, linewidth=1.0 )
            axes[i].scatter( x[error_lines[i]], y[error_lines[i],i], **reference_style )
            axes[i].scatter( x[error_lines[i]], y_pred[error_lines[i],i], **prediction_style )
    ## plot decoration
    n_top = 4 #bottom ticks are always 2
    n_bot = 1
    for i in range( n_target):
        ylim = 1.05* np.abs(y[:,i]).max()
        bar_tick_incr = max_errors[i,:].max()/n_bot
        if i < 2: 
            top_ticks = np.arange( 0, ylim, (ylim+ylim/n_top)/n_top)
            bottom_ticks = np.arange( -max_errors[i,:].max(), 1e-5, bar_tick_incr )
            yticks = np.around( np.hstack( (bottom_ticks, top_ticks) ), 2)
            ymin = max( -1.35*np.abs(bottom_ticks).max(), -0.5*ylim )
            axes[i].set_yticks( yticks)
            axes[i].set_yticklabels( np.abs(yticks) )
        else: #extra treatment of k12
            y_incr = 2*ylim/(2*n_top)
            ymin = -ylim-max_errors[i].max()
            bar_ticks = np.arange(-max_errors[i,:].max(), bar_tick_incr, bar_tick_incr ) 
            bar_ticks = bar_ticks + bar_kwargs['bottom']
            yticks = np.around( np.arange( -ylim+y_incr, ylim, y_incr ), 2)
            yticklabels =  list( np.abs(np.around( bar_ticks-bar_kwargs['bottom'],2)))+ list(yticks)  
            yticks =  list( bar_ticks) + list(yticks)
            axes[i].axhline( bar_kwargs['bottom'] , color='black' )
            axes[i].set_yticks( yticks)
            axes[i].set_yticklabels( yticklabels )
        axes[i].set_ylabel( ylabels[i])
        axes[i].set_xlabel( 'volume fraction [-]')
        if quantiles:
            axes[i].set_ylim(ymin=ymin, ymax=ylim)
        if not axes[i]._gridOn:
            axes[i].grid()
    axes[1].set_zorder( 1)
    if extra_legend and quantiles:
        label_cmap = Cmap.from_list( '', quantile_colors[:len(quantiles)+1][::-1] )
        legend_handler = plot.HandlerColormap( label_cmap, quantile_label, n_stripes=len(quantiles) + 1 )
        top_ylim = axes[0].get_ylim()[1]
        axes[0].set_ylim( ymax=1e10 )
        legend = plot.add_legend( axes[0], **legend_handler.get_legend_entries( axes[0]) )
        plot.export_legend( legend, extra_legend )
        legend.remove()
        axes[0].set_ylim( ymax=top_ylim) 
    else:
        legend = axes[1].legend( bbox_to_anchor=(0.65,1.1) )
        legend.get_frame().set_alpha(None)
    return fig, axes



